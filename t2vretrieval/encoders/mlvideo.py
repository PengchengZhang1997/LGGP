import torch.nn as nn
from t2vretrieval.encoders.transformer_legacy import TransformerLegacy
import framework.configbase
import torch
import t2vretrieval.encoders.graph
import numpy as np
from t2vretrieval.encoders.discriminator import Discriminator
from t2vretrieval.encoders.readout import AvgReadout

class MultilevelEncoderConfig(framework.configbase.ModuleConfig):
  def __init__(self):
    super().__init__()
    self.dim_fts = [2048]
    self.dim_embed = 512
    self.dropout = 0
    self.num_levels = 4
    self.share_enc = False
    self.node_num = 27

class MultilevelEncoder(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    input_size = sum(self.config.dim_fts)
    self.dropout = nn.Dropout(self.config.dropout)
    self.mot_wei = nn.Linear(input_size, 1)
    self.mot_fc = nn.Linear(input_size, self.config.dim_embed)
    self.app_fc = nn.Linear(input_size, self.config.dim_embed)
    self.activate = nn.GELU()
    self.transformer_local = TransformerLegacy(input_size, self.config.dim_embed)
    self.transformer_global = TransformerLegacy(self.config.dim_embed, self.config.dim_embed, ispooling=False)
    #self.mot_attn = nn.Linear(self.config.dim_embed, 1, bias=True)
    #self.app_attn = nn.Linear(self.config.dim_embed, 1, bias=True)
    # GCN parameters
    self.gcn = t2vretrieval.encoders.graph.GCNEncoder(self.config.dim_embed, self.config.dim_embed, 1,
                                                      attention=True, embed_first=False, dropout=0.2)
    # GCN-global parameters
    self.gcn_global = t2vretrieval.encoders.graph.GCNEncoder(self.config.dim_embed, self.config.dim_embed, 1,
                                                             attention=False, embed_first=False, dropout=0.2)
    # pooling
    self.gcn_pooling = t2vretrieval.encoders.graph.GCNEncoder(self.config.dim_embed, self.config.dim_embed, 1,
                                                              attention=False, embed_first=False, dropout=0.2)
    self.pooling_ft = nn.Linear(self.config.dim_embed, 1, bias=True)
    self.disc = Discriminator(self.config.dim_embed)
    self.disc_global = Discriminator(self.config.dim_embed)
    self.sigm = nn.Sigmoid()
    self.info_loss = nn.BCEWithLogitsLoss()
      
  def forward(self, vid_ft, vid_len, clip_ft, clip_lens, seg_nums, clip_edges, gedges):
    '''
    inputs:
       vid_ft: (btach_size, vid_len, dim),
       vid_len: (batch_size),
       clip_ft: (batch_size, seg_num, clip_len, dim),
       clip_len: (batch_size, seg_num),
       seg_num: (batch_size)
    outputs:
       globel_embeds: (batch_size, vid_len, new_dim),
       segment_embeds: (batch_size, seg_num, new_dim),
       motion_embeds: (batch_size, seg_num, clip_len, new_dim),
       appearance_embeds: (batch_size, seg_num, clip_len, new_dim)
    '''

    batch_size, vlen, _ = vid_ft.shape
    hidden_dim = self.config.dim_embed
    clen = clip_ft.shape[1]
    max_seg_len = max(seg_nums)
    max_dataclip_len = self.config.node_num

    # global level
    input_pad_masks1 = framework.ops.sequence_mask(vid_len, max_len=vlen, inverse=True)
    global_embeds, _ = self.transformer_local(vid_ft, input_pad_masks1, vid_len)

    # segment_level
    input_pad_masks2 = framework.ops.sequence_mask(clip_lens, max_len=clen, inverse=True)
    segment_embeds1, _ = self.transformer_local(clip_ft, input_pad_masks2, clip_lens)

    # motion level
    mot_scores = self.mot_wei(clip_ft).squeeze(2)
    input_pad_masks = framework.ops.sequence_mask(clip_lens,
                                                  max_len=mot_scores.size(1), inverse=True)
    attn_scores = mot_scores.masked_fill(input_pad_masks, -1e18)
    vid_clip = []
    for i in range(clen-2):
      clip_ft_now = clip_ft[:, i:i+3]
      attn_score = attn_scores[:, i:i+3]
      attn_score = torch.softmax(attn_score, dim=1)
      sent_embeds = torch.sum(clip_ft_now * attn_score.unsqueeze(2), 1)
      vid_clip.append(sent_embeds)
    vid_clip = torch.stack(vid_clip).permute(1, 0, 2)
    motion_embeds1 = self.activate(self.mot_fc(vid_clip))
    # object level
    appearance_embeds1 = self.activate(self.app_fc(clip_ft))
    # clip_gcn
    node_embeds1 = torch.cat([segment_embeds1.unsqueeze(1), motion_embeds1, appearance_embeds1], 1)

    # construction of local negative examples
    idx_local_mot = np.random.permutation(motion_embeds1.shape[1])
    idx_local_obj = np.random.permutation(appearance_embeds1.shape[1])
    motion_embeds2 = motion_embeds1[:, idx_local_mot, :]
    appearance_embeds2 = appearance_embeds1[:, idx_local_obj, :]
    node_embeds_neg = torch.cat([segment_embeds1.unsqueeze(1), motion_embeds2, appearance_embeds2], 1)

    node_embeds2 = self.gcn(node_embeds1, clip_edges)
    segment_embeds2 = node_embeds2[:, 0]
    motion_embeds2 = node_embeds2[:, 1: 19].contiguous()
    appearance_embeds2 = node_embeds2[:, 19:].contiguous()

    node_embeds_neg2 = self.gcn(node_embeds_neg, clip_edges)
    motion_embeds_neg2 = node_embeds_neg2[:, 1: 19].contiguous()
    appearance_embeds_neg2 = node_embeds_neg2[:, 19:].contiguous()

    # hierechical attention graph pooling
    node_embeds3 = self.gcn_pooling(node_embeds2, clip_edges)
    node_embeds = self.activate(self.pooling_ft(node_embeds3)).squeeze(2)

    # motion level pooling
    attn_scores1 = node_embeds[:, 1: 19].contiguous()
    input_pad_masks = framework.ops.sequence_mask(clip_lens-2, max_len=attn_scores1.size(1), inverse=True)
    attn_scores2 = attn_scores1.masked_fill(input_pad_masks, -1e18)
    attn_scores3 = torch.softmax(attn_scores2, dim=1)
    motion_embeds3 = torch.sum(motion_embeds2 * attn_scores3.unsqueeze(2), 1)

    # appearance level pooling
    attn_scores4 = node_embeds[:, 19:].contiguous()
    input_pad_masks = framework.ops.sequence_mask(clip_lens, max_len=attn_scores4.size(1), inverse=True)
    attn_scores5 = attn_scores4.masked_fill(input_pad_masks, -1e18)
    attn_scores6 = torch.softmax(attn_scores5, dim=1)
    appearance_embeds3 = torch.sum(appearance_embeds2 * attn_scores6.unsqueeze(2), 1)

    c_local_mot = self.sigm(motion_embeds3)
    c_local_obj = self.sigm(appearance_embeds3)

    ret_local_mot = self.disc(c_local_mot, motion_embeds2, motion_embeds_neg2, None, None)
    ret_local_obj = self.disc(c_local_obj, appearance_embeds2, appearance_embeds_neg2, None, None)

    lbl_1 = torch.ones(clip_ft.shape[0], motion_embeds1.shape[1])
    lbl_2 = torch.zeros(clip_ft.shape[0], motion_embeds1.shape[1])
    lbl = torch.cat((lbl_1, lbl_2), 1).to(clip_ft.device)
    ret_local_mot_loss = self.info_loss(ret_local_mot, lbl)

    lbl_1 = torch.ones(clip_ft.shape[0], appearance_embeds1.shape[1])
    lbl_2 = torch.zeros(clip_ft.shape[0], appearance_embeds1.shape[1])
    lbl = torch.cat((lbl_1, lbl_2), 1).to(clip_ft.device)
    ret_local_obj_loss = self.info_loss(ret_local_obj, lbl)

    seg_emb_reshape1 = torch.zeros((batch_size, max_seg_len, hidden_dim)).float().to(segment_embeds2.device)
    mot_emb_reshape1 = torch.zeros((batch_size, max_seg_len, hidden_dim)).float().to(segment_embeds2.device)
    app_emb_reshape1 = torch.zeros((batch_size, max_seg_len, hidden_dim)).float().to(segment_embeds2.device)
    seg_emb_mask = torch.ones((batch_size, max_seg_len)).bool().to(segment_embeds2.device)
    sent_emb_lens = torch.zeros((batch_size,)).long().to(segment_embeds2.device)

    pointer = 0
    for batch_num, clip_len in enumerate(seg_nums):
      seg_emb_reshape1[batch_num, :clip_len, :] = segment_embeds2[pointer:pointer + clip_len, :]
      mot_emb_reshape1[batch_num, :clip_len, :] = motion_embeds3[pointer:pointer + clip_len, :]
      app_emb_reshape1[batch_num, :clip_len, :] = appearance_embeds3[pointer:pointer + clip_len, :]
      seg_emb_mask[batch_num, :clip_len] = 0
      sent_emb_lens[batch_num] = clip_len
      pointer += clip_len

    seg_emb_reshape2 = self.transformer_global(seg_emb_reshape1, seg_emb_mask, sent_emb_lens)
    mot_emb_reshape2 = self.transformer_global(mot_emb_reshape1, seg_emb_mask, sent_emb_lens)
    app_emb_reshape2 = self.transformer_global(app_emb_reshape1, seg_emb_mask, sent_emb_lens)

    new_emb = torch.zeros((batch_size, max_dataclip_len, hidden_dim)).float().to(segment_embeds2.device)
    new_emb[:, :max_seg_len] = seg_emb_reshape2
    seg_emb_reshape3 = new_emb
    new_emb[:, :max_seg_len] = mot_emb_reshape2
    mot_emb_reshape3 = new_emb
    new_emb[:, :max_seg_len] = app_emb_reshape2
    app_emb_reshape3 = new_emb

    # global gcn
    new_emb = torch.cat([global_embeds.unsqueeze(1), seg_emb_reshape3, mot_emb_reshape3, app_emb_reshape3], 1)

    # construction of global negative examples
    idx_global_seg = np.random.permutation(seg_emb_reshape3.shape[1])
    idx_global_mot = np.random.permutation(mot_emb_reshape3.shape[1])
    idx_global_obj = np.random.permutation(app_emb_reshape3.shape[1])
    seg_neg = seg_emb_reshape3[:, idx_global_seg, :]
    mot_neg = seg_emb_reshape3[:, idx_global_mot, :]
    obj_neg = seg_emb_reshape3[:, idx_global_obj, :]
    new_emb_neg = torch.cat([global_embeds.unsqueeze(1), seg_neg, mot_neg, obj_neg], 1)

    new_emb = self.gcn_global(new_emb, gedges)
    global_embeds2 = new_emb[:, 0]
    seg_emb_reshape4 = new_emb[:, 1: 1 + max_dataclip_len].contiguous()
    mot_emb_reshape4 = new_emb[:, 1 + max_dataclip_len: 1 + max_dataclip_len * 2].contiguous()
    app_emb_reshape4 = new_emb[:, 1 + max_dataclip_len * 2:].contiguous()
    seg_emb_reshape5 = seg_emb_reshape4[:, :max_seg_len].contiguous()
    mot_emb_reshape5 = mot_emb_reshape4[:, :max_seg_len].contiguous()
    app_emb_reshape5 = app_emb_reshape4[:, :max_seg_len].contiguous()

    new_neg = self.gcn_global(new_emb_neg, gedges)
    seg_emb_neg = new_neg[:, 1: 1 + max_dataclip_len].contiguous()
    mot_emb_neg = new_neg[:, 1 + max_dataclip_len: 1 + max_dataclip_len*2].contiguous()
    obj_emb_neg = new_neg[:, 1 + max_dataclip_len*2:].contiguous()

    len_div = sent_emb_lens.unsqueeze(-1).float()
    seg_emb_reshape = torch.sum(seg_emb_reshape5, dim=1) / len_div
    mot_emb_reshape = torch.sum(mot_emb_reshape5, dim=1) / len_div
    app_emb_reshape = torch.sum(app_emb_reshape5, dim=1) / len_div
    fusion_embeds = torch.cat([global_embeds2, seg_emb_reshape, mot_emb_reshape, app_emb_reshape], dim=-1)
    #fusion_embeds = torch.cat([global_embeds2, seg_emb_reshape, app_emb_reshape], dim=-1)

    c_seg_global = self.sigm(seg_emb_reshape)
    c_mot_global = self.sigm(mot_emb_reshape)
    c_obj_global = self.sigm(app_emb_reshape)
    ret_global_seg = self.disc_global(c_seg_global, seg_emb_reshape4, seg_emb_neg, None, None)
    ret_global_mot = self.disc_global(c_mot_global, mot_emb_reshape4, mot_emb_neg, None, None)
    ret_global_obj = self.disc_global(c_obj_global, app_emb_reshape4, obj_emb_neg, None, None)

    lbl_1 = torch.ones(batch_size, seg_emb_reshape3.shape[1])
    lbl_2 = torch.zeros(batch_size, seg_emb_reshape3.shape[1])
    lbl = torch.cat((lbl_1, lbl_2), 1).to(clip_ft.device)
    ret_global_seg_loss = self.info_loss(ret_global_seg, lbl)

    lbl_1 = torch.ones(batch_size, mot_emb_reshape3.shape[1])
    lbl_2 = torch.zeros(batch_size, mot_emb_reshape3.shape[1])
    lbl = torch.cat((lbl_1, lbl_2), 1).to(clip_ft.device)
    ret_global_mot_loss = self.info_loss(ret_global_mot, lbl)

    lbl_1 = torch.ones(batch_size, app_emb_reshape3.shape[1])
    lbl_2 = torch.zeros(batch_size, app_emb_reshape3.shape[1])
    lbl = torch.cat((lbl_1, lbl_2), 1).to(clip_ft.device)
    ret_global_obj_loss = self.info_loss(ret_global_obj, lbl)

    return fusion_embeds, global_embeds2, segment_embeds2, motion_embeds3, appearance_embeds3,\
           ret_local_mot_loss, ret_local_obj_loss, ret_global_seg_loss, ret_global_mot_loss, ret_global_obj_loss

import torch
import torch.nn as nn
from t2vretrieval.encoders.transformer_legacy import TransformerLegacy
import framework.configbase
import framework.ops
import t2vretrieval.encoders.graph
import t2vretrieval.encoders.sentence
import numpy as np
from t2vretrieval.encoders.discriminator import Discriminator
from t2vretrieval.encoders.readout import AvgReadout

INF = 32752

class RoleGraphEncoderConfig(framework.configbase.ModuleConfig):
  def __init__(self):
    super().__init__()
    self.num_roles = 16
    self.num_words = 0
    self.dim_word = 1536
    self.bidirectional = True
    self.rnn_hidden_size = 512
    self.num_layers = 1
    self.dropout = 0.5
    self.gcn_num_layers = 1
    self.gcn_attention = False
    self.gcn_dropout = 0.5
    self.node_num = 27

class RoleGraphEncoder(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.transformer_local = TransformerLegacy(self.config.dim_word, self.config.rnn_hidden_size)
    self.transformer_global = TransformerLegacy(self.config.rnn_hidden_size, self.config.rnn_hidden_size, ispooling=False)
    if self.config.num_roles > 0:
      self.role_embedding = nn.Embedding(self.config.num_roles, self.config.rnn_hidden_size)

    # GCN parameters
    self.gcn = t2vretrieval.encoders.graph.GCNEncoder(self.config.rnn_hidden_size,
                                                      self.config.rnn_hidden_size, self.config.gcn_num_layers,
                                                      attention=self.config.gcn_attention,
                                                      embed_first=False, dropout=0.2)
    # GCN-global parameters
    self.gcn_global = t2vretrieval.encoders.graph.GCNEncoder(self.config.rnn_hidden_size,
                                                      self.config.rnn_hidden_size, self.config.gcn_num_layers,
                                                      attention=False, embed_first=False, dropout=0.2)

    # for pooling
    self.gcn_pooling = t2vretrieval.encoders.graph.GCNEncoder(self.config.rnn_hidden_size,
                                                             self.config.rnn_hidden_size, self.config.gcn_num_layers,
                                                             attention=False, embed_first=False, dropout=0.2)
    self.pooling_ft = nn.Linear(self.config.rnn_hidden_size, 1, bias=True)
    #self.verb_attn = nn.Linear(self.config.rnn_hidden_size, 1, bias=True)
    #self.noun_attn = nn.Linear(self.config.rnn_hidden_size, 1, bias=True)
    self.activate = nn.GELU()
    self.activate = nn.GELU()
    self.disc = Discriminator(self.config.rnn_hidden_size)
    self.disc_global = Discriminator(self.config.rnn_hidden_size)
    self.sigm = nn.Sigmoid()
    self.info_loss = nn.BCEWithLogitsLoss()

  def pool_phrases(self, word_embeds, phrase_masks, pool_type='avg'):
    '''
    Args:
      word_embeds: (batch, max_sent_len, embed_size)
      phrase_masks: (batch, num_phrases, max_sent_len)
    Returns:
      phrase_embeds: (batch, num_phrases, embed_size)
    '''
    if pool_type == 'avg':
      # (batch, num_phrases, max_sent_len, embed_size)
      phrase_masks = phrase_masks.float()
      phrase_embeds = torch.bmm(phrase_masks, word_embeds) / torch.sum(phrase_masks, 2, keepdim=True).clamp(min=1)
    elif pool_type == 'max':
      embeds = word_embeds.unsqueeze(1).masked_fill(phrase_masks.unsqueeze(3)==0, -INF)
      phrase_embeds = torch.max(embeds, 2)[0]
    else:
      raise NotImplementedError
    return phrase_embeds

  def forward(self, para_embs, para_lens, sent_embs, sent_lens, verb_masks, noun_masks, node_roles, rel_edges, sent_nums, gedges):
    '''
    Args:
      para_emb: (batch, max_sent_len_para, dim)
      para_lens: (batch, )
      sent_emb: (batch, seg_num, max_sent_len, dim)
      sent_lens: (batch, seg_num)
      verb_masks: (batch, seg_num, num_verbs, max_sent_len)
      noun_masks: (batch, seg_num, num_nouns, max_sent_len)
      node_roles: (batch, seg_num, num_verbs + num_nouns)
    '''
    batch_size, plen, _ = para_embs.shape
    hidden_dim = self.config.rnn_hidden_size
    slen = sent_embs.shape[1]
    max_sent_len = max(sent_nums)
    max_dataclip_len = self.config.node_num

    # paragraph level
    input_pad_masks1 = framework.ops.sequence_mask(para_lens, max_len=plen, inverse=True)
    para_embeds1, para_word_emb = self.transformer_local(para_embs, input_pad_masks1, para_lens)

    # other levels
    input_pad_masks2 = framework.ops.sequence_mask(sent_lens, max_len=slen, inverse=True)
    sent_ctx_embeds1, word_embeds = self.transformer_local(sent_embs, input_pad_masks2, sent_lens)

    num_verbs = verb_masks.size(1)
    verb_embeds = self.pool_phrases(word_embeds, verb_masks, pool_type='max')
    if self.config.num_roles > 0:
      verb_ctx_embeds1 = verb_embeds * self.role_embedding(node_roles[:, :num_verbs])
    num_nouns = noun_masks.size(1)
    noun_embeds = self.pool_phrases(word_embeds, noun_masks, pool_type='max')
    if self.config.num_roles > 0:
      noun_ctx_embeds1 = noun_embeds * self.role_embedding(node_roles[:, num_verbs:])

    node_ctx_embeds2 = torch.cat([sent_ctx_embeds1.unsqueeze(1), verb_ctx_embeds1, noun_ctx_embeds1], 1)
    node_ctx_embeds3 = self.gcn(node_ctx_embeds2, rel_edges)

    # construction of local negative examples
    idx_local_mot = np.random.permutation(verb_ctx_embeds1.shape[1])
    idx_local_obj = np.random.permutation(noun_ctx_embeds1.shape[1])
    motion_embeds2 = verb_ctx_embeds1[:, idx_local_mot, :]
    appearance_embeds2 = noun_ctx_embeds1[:, idx_local_obj, :]
    node_embeds_neg = torch.cat([sent_ctx_embeds1.unsqueeze(1), motion_embeds2, appearance_embeds2], 1)
    node_ctx_neg = self.gcn(node_embeds_neg, rel_edges)

    sent_ctx_embeds2 = node_ctx_embeds3[:, 0]
    verb_ctx_embeds2 = node_ctx_embeds3[:, 1: 1 + num_verbs].contiguous()
    noun_ctx_embeds2 = node_ctx_embeds3[:, 1 + num_verbs:].contiguous()

    verb_neg = node_ctx_neg[:, 1: 1 + num_verbs].contiguous()
    noun_neg = node_ctx_neg[:, 1 + num_verbs:].contiguous()

    # hierarchical attention graph pooling
    node_ctx_embeds4 = self.gcn_pooling(node_ctx_embeds3, rel_edges)
    node_ctx_embeds = self.activate(self.pooling_ft(node_ctx_embeds4)).squeeze(2)

    # verb level pooling
    attn_scores1 = node_ctx_embeds[:, 1: 1 + num_verbs].contiguous()
    verb_lens = torch.sum(verb_masks, 2)
    input_pad_masks3 = framework.ops.sequence_mask(torch.sum(verb_lens > 0, 1).long(), max_len=num_verbs, inverse=True)
    attn_scores2 = attn_scores1.masked_fill(input_pad_masks3, -1e18)
    attn_scores3 = torch.softmax(attn_scores2, dim=1)
    verb_ctx_embeds = torch.sum(verb_ctx_embeds2 * attn_scores3.unsqueeze(2), 1)

    # noun level pooling
    noun_lens = torch.sum(noun_masks, 2)
    attn_scores4 = node_ctx_embeds[:, 1 + num_verbs:].contiguous()
    input_pad_masks4 = framework.ops.sequence_mask(torch.sum(noun_lens > 0, 1).long(), max_len=num_nouns, inverse=True)
    attn_scores5 = attn_scores4.masked_fill(input_pad_masks4, -1e18)
    attn_scores6 = torch.softmax(attn_scores5, dim=1)
    noun_ctx_embeds = torch.sum(noun_ctx_embeds2 * attn_scores6.unsqueeze(2), 1)

    c_local_mot = self.sigm(verb_ctx_embeds)
    c_local_obj = self.sigm(noun_ctx_embeds)
    ret_local_mot = self.disc(c_local_mot, verb_ctx_embeds2, verb_neg, None, None)
    ret_local_obj = self.disc(c_local_obj, noun_ctx_embeds2, noun_neg, None, None)

    lbl_1 = torch.ones(sent_embs.shape[0], verb_ctx_embeds1.shape[1])
    lbl_2 = torch.zeros(sent_embs.shape[0], verb_ctx_embeds1.shape[1])
    lbl = torch.cat((lbl_1, lbl_2), 1).to(sent_embs.device)
    ret_local_mot_loss = self.info_loss(ret_local_mot, lbl)

    lbl_1 = torch.ones(sent_embs.shape[0], noun_ctx_embeds1.shape[1])
    lbl_2 = torch.zeros(sent_embs.shape[0], noun_ctx_embeds1.shape[1])
    lbl = torch.cat((lbl_1, lbl_2), 1).to(sent_embs.device)
    ret_local_obj_loss = self.info_loss(ret_local_obj, lbl)

    sent_emb_reshape1 = torch.zeros((batch_size, max_sent_len, hidden_dim)).float().to(sent_embs.device)
    verb_emb_reshape1 = torch.zeros((batch_size, max_sent_len, hidden_dim)).float().to(sent_embs.device)
    noun_emb_reshape1 = torch.zeros((batch_size, max_sent_len, hidden_dim)).float().to(sent_embs.device)
    sent_emb_mask = torch.ones((batch_size, max_sent_len)).bool().to(sent_embs.device)
    sent_emb_lens = torch.zeros((batch_size,)).long().to(sent_embs.device)

    pointer = 0
    for batch_num, clip_len in enumerate(sent_nums):
      sent_emb_reshape1[batch_num, :clip_len, :] = sent_ctx_embeds2[pointer:pointer + clip_len, :]
      verb_emb_reshape1[batch_num, :clip_len, :] = verb_ctx_embeds[pointer:pointer + clip_len, :]
      noun_emb_reshape1[batch_num, :clip_len, :] = noun_ctx_embeds[pointer:pointer + clip_len, :]
      sent_emb_mask[batch_num, :clip_len] = 0
      sent_emb_lens[batch_num] = clip_len
      pointer += clip_len

    sent_emb_reshape2 = self.transformer_global(sent_emb_reshape1, sent_emb_mask, sent_emb_lens)
    verb_emb_reshape2 = self.transformer_global(verb_emb_reshape1, sent_emb_mask, sent_emb_lens)
    noun_emb_reshape2 = self.transformer_global(noun_emb_reshape1, sent_emb_mask, sent_emb_lens)

    new_emb = torch.zeros((batch_size, max_dataclip_len, hidden_dim)).float().to(sent_embs.device)
    new_emb[:, :max_sent_len] = sent_emb_reshape2
    sent_emb_reshape3 = new_emb
    new_emb[:, :max_sent_len] = verb_emb_reshape2
    verb_emb_reshape3 = new_emb
    new_emb[:, :max_sent_len] = noun_emb_reshape2
    noun_emb_reshape3 = new_emb

    # global gcn
    new_emb = torch.cat([para_embeds1.unsqueeze(1), sent_emb_reshape3, verb_emb_reshape3, noun_emb_reshape3], 1)
    new_emb = self.gcn_global(new_emb, gedges)

    # construction of global negative examples
    idx_global_seg = np.random.permutation(sent_emb_reshape3.shape[1])
    idx_global_mot = np.random.permutation(verb_emb_reshape3.shape[1])
    idx_global_obj = np.random.permutation(noun_emb_reshape3.shape[1])
    seg_neg = sent_emb_reshape3[:, idx_global_seg, :]
    mot_neg = verb_emb_reshape3[:, idx_global_mot, :]
    obj_neg = noun_emb_reshape3[:, idx_global_obj, :]
    new_emb_neg = torch.cat([para_embeds1.unsqueeze(1), seg_neg, mot_neg, obj_neg], 1)

    para_embeds = new_emb[:, 0]
    sent_emb_reshape4 = new_emb[:, 1: 1 + max_dataclip_len].contiguous()
    verb_emb_reshape4 = new_emb[:, 1 + max_dataclip_len: 1 + max_dataclip_len * 2].contiguous()
    noun_emb_reshape4 = new_emb[:, 1 + max_dataclip_len * 2:].contiguous()
    sent_emb_reshape5 = sent_emb_reshape4[:, :max_sent_len].contiguous()
    verb_emb_reshape5 = verb_emb_reshape4[:, :max_sent_len].contiguous()
    noun_emb_reshape5 = noun_emb_reshape4[:, :max_sent_len].contiguous()

    new_neg = self.gcn_global(new_emb_neg, gedges)
    seg_emb_neg = new_neg[:, 1: 1 + max_dataclip_len].contiguous()
    mot_emb_neg = new_neg[:, 1 + max_dataclip_len: 1 + max_dataclip_len * 2].contiguous()
    obj_emb_neg = new_neg[:, 1 + max_dataclip_len * 2:].contiguous()

    len_div = sent_emb_lens.unsqueeze(-1).float()
    sent_emb_reshape = torch.sum(sent_emb_reshape5, dim=1) / len_div
    verb_emb_reshape = torch.sum(verb_emb_reshape5, dim=1) / len_div
    noun_emb_reshape = torch.sum(noun_emb_reshape5, dim=1) / len_div
    fusion_embeds = torch.cat([para_embeds, sent_emb_reshape, verb_emb_reshape, noun_emb_reshape], dim=-1)
    #fusion_embeds = torch.cat([para_embeds, sent_emb_reshape, noun_emb_reshape], dim=-1)

    c_seg_global = self.sigm(sent_emb_reshape)
    c_mot_global = self.sigm(verb_emb_reshape)
    c_obj_global = self.sigm(noun_emb_reshape)
    ret_global_seg = self.disc_global(c_seg_global, sent_emb_reshape4, seg_emb_neg, None, None)
    ret_global_mot = self.disc_global(c_mot_global, verb_emb_reshape4, mot_emb_neg, None, None)
    ret_global_obj = self.disc_global(c_obj_global, noun_emb_reshape4, obj_emb_neg, None, None)

    lbl_1 = torch.ones(batch_size, sent_emb_reshape3.shape[1])
    lbl_2 = torch.zeros(batch_size, sent_emb_reshape3.shape[1])
    lbl = torch.cat((lbl_1, lbl_2), 1).to(sent_embs.device)
    ret_global_seg_loss = self.info_loss(ret_global_seg, lbl)

    lbl_1 = torch.ones(batch_size, verb_emb_reshape3.shape[1])
    lbl_2 = torch.zeros(batch_size, verb_emb_reshape3.shape[1])
    lbl = torch.cat((lbl_1, lbl_2), 1).to(sent_embs.device)
    ret_global_mot_loss = self.info_loss(ret_global_mot, lbl)

    lbl_1 = torch.ones(batch_size, noun_emb_reshape3.shape[1])
    lbl_2 = torch.zeros(batch_size, noun_emb_reshape3.shape[1])
    lbl = torch.cat((lbl_1, lbl_2), 1).to(sent_embs.device)
    ret_global_obj_loss = self.info_loss(ret_global_obj, lbl)

    return fusion_embeds, para_embeds, sent_ctx_embeds2, verb_ctx_embeds, noun_ctx_embeds,\
           ret_local_mot_loss, ret_local_obj_loss, ret_global_seg_loss, ret_global_mot_loss, ret_global_obj_loss

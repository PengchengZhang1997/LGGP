import torch
import t2vretrieval.encoders.mlsent
import t2vretrieval.encoders.mlvideo
import t2vretrieval.models.globalmatch
from t2vretrieval.models.criterion import cosine_sim
from t2vretrieval.models.globalmatch import VISENC, TXTENC
import torch.nn as nn

INF = 32752

class RoleGraphMatchModelConfig(t2vretrieval.models.globalmatch.GlobalMatchModelConfig):
    def __init__(self):
        super().__init__()
        self.num_verbs = 4
        self.num_nouns = 6

        self.attn_fusion = 'embed'  # sim, embed
        self.simattn_sigma = 4

        self.hard_topk = 1
        self.max_violation = True
        self.loss_weights = None

        self.subcfgs[VISENC] = t2vretrieval.encoders.mlvideo.MultilevelEncoderConfig()
        self.subcfgs[TXTENC] = t2vretrieval.encoders.mlsent.RoleGraphEncoderConfig()


class RoleGraphMatchModel(t2vretrieval.models.globalmatch.GlobalMatchModel):
    def build_submods(self):
        return {
            VISENC: t2vretrieval.encoders.mlvideo.MultilevelEncoder(self.config.subcfgs[VISENC]),
            TXTENC: t2vretrieval.encoders.mlsent.RoleGraphEncoder(self.config.subcfgs[TXTENC])
        }

    def forward_video_embed(self, batch_data):
        vid_fts = torch.FloatTensor(batch_data['vid_fts']).to(self.device)
        vid_lens = torch.LongTensor(batch_data['vid_lens']).to(self.device)
        clip_fts = torch.FloatTensor(batch_data['clip_fts']).to(self.device)
        clip_lens = torch.LongTensor(batch_data['clip_lens']).to(self.device)
        seg_nums = torch.LongTensor(batch_data['seg_nums']).to(self.device)
        clip_edges = torch.FloatTensor(batch_data['vid_edges']).to(self.device)
        gedges = torch.FloatTensor(batch_data['gedges']).to(self.device)

        video_embeds, global_embeds, clip_embeds, mot_embeds, obj_embeds,\
        ret_local_mot, ret_local_obj, ret_global_seg, ret_global_mot, ret_global_obj = self.submods[VISENC](
            vid_fts, vid_lens, clip_fts, clip_lens, seg_nums, clip_edges, gedges)

        return {
            'video_embeds': video_embeds,
            'global_vid_embeds': global_embeds,
            'clip_vid_embeds': clip_embeds,
            'mot_vid_embeds': mot_embeds,
            'obj_vid_embeds': obj_embeds,
            'vret_lm': ret_local_mot,
            'vret_lo': ret_local_obj,
            'vret_gs': ret_global_seg,
            'vret_gm': ret_global_mot,
            'vret_go': ret_global_obj,
        }

    def forward_text_embed(self, batch_data):
        para_embs = torch.FloatTensor(batch_data['para_embs']).to(self.device)
        para_lens = torch.LongTensor(batch_data['para_lens']).to(self.device)
        sent_embs = torch.FloatTensor(batch_data['sent_embs']).to(self.device)
        sent_lens = torch.LongTensor(batch_data['sent_lens']).to(self.device)
        verb_masks = torch.BoolTensor(batch_data['verb_masks']).to(self.device)
        noun_masks = torch.BoolTensor(batch_data['noun_masks']).to(self.device)
        node_roles = torch.LongTensor(batch_data['node_roles']).to(self.device)
        rel_edges = torch.FloatTensor(batch_data['rel_edges']).to(self.device)
        sent_nums = torch.LongTensor(batch_data['sent_nums']).to(self.device)
        gedges = torch.FloatTensor(batch_data['gedges']).to(self.device)

        para_embeds, global_embeds, clip_embeds, mot_embeds, obj_embeds,\
        ret_local_mot, ret_local_obj, ret_global_seg, ret_global_mot, ret_global_obj = self.submods[TXTENC](
            para_embs, para_lens, sent_embs, sent_lens, verb_masks, noun_masks, node_roles, rel_edges, sent_nums, gedges)

        return {
            'para_embeds': para_embeds,
            'global_embeds': global_embeds,
            'clip_embeds': clip_embeds,
            'mot_embeds': mot_embeds,
            'obj_embeds': obj_embeds,
            'tret_lm': ret_local_mot,
            'tret_lo': ret_local_obj,
            'tret_gs': ret_global_seg,
            'tret_gm': ret_global_mot,
            'tret_go': ret_global_obj,
        }

    def generate_scores(self, im, s):
        scores = cosine_sim(im, s)
        return scores

    def forward_loss(self, batch_data, step=None):
        vid_enc_outs = self.forward_video_embed(batch_data)
        cap_enc_outs = self.forward_text_embed(batch_data)
        vid_enc_outs.update(cap_enc_outs)
        scores = self.generate_scores(vid_enc_outs['video_embeds'], vid_enc_outs['para_embeds'])
        #align loss
        high_align_loss = self.criterion(vid_enc_outs['video_embeds'], vid_enc_outs['para_embeds'])
        low_align_loss = self.criterion(vid_enc_outs['clip_vid_embeds'], vid_enc_outs['clip_embeds']) \
                         + self.criterion(vid_enc_outs['mot_vid_embeds'], vid_enc_outs['mot_embeds']) \
                         + self.criterion(vid_enc_outs['obj_vid_embeds'], vid_enc_outs['obj_embeds'])

        glo_align_loss = self.criterion(vid_enc_outs['global_vid_embeds'], vid_enc_outs['global_embeds'])
        #cluster loss
        high_cluster_loss = (self.criterion(vid_enc_outs['video_embeds'],
                                            vid_enc_outs['video_embeds']) + self.criterion(
            vid_enc_outs['para_embeds'], vid_enc_outs['para_embeds'])) / 2
        low_cluster_loss = (self.criterion(vid_enc_outs['clip_vid_embeds'],
                                           vid_enc_outs['clip_vid_embeds'],) + self.criterion(
            vid_enc_outs['clip_embeds'], vid_enc_outs['clip_embeds'])) / 2
        glo_cluster_loss = (self.criterion(vid_enc_outs['global_vid_embeds'],
                                           vid_enc_outs['global_vid_embeds']) + self.criterion(
            vid_enc_outs['global_embeds'], vid_enc_outs['global_embeds'])) / 2

        #InfoMax loss
        infomax_loss_text = (vid_enc_outs['tret_lm'] + vid_enc_outs['tret_lo'] + vid_enc_outs['tret_gs']+ vid_enc_outs[
            'tret_gm'] + vid_enc_outs['tret_go']) / 10
        infomax_loss_vid = (vid_enc_outs['vret_lm'] + vid_enc_outs['vret_lo'] + vid_enc_outs['vret_gs'] + vid_enc_outs[
            'vret_gm'] + vid_enc_outs['vret_go']) / 10

        loss = high_align_loss + low_align_loss + glo_align_loss + infomax_loss_text + infomax_loss_vid# + high_cluster_loss + low_cluster_loss
        #loss = high_align_loss + low_align_loss + glo_align_loss + fus_align_loss + high_cluster_loss +low_cluster_loss + glo_cluster_loss + fus_cluster_loss

        if step is not None and self.config.monitor_iter > 0 and step % self.config.monitor_iter == 0:
            neg_scores = scores.masked_fill(torch.eye(len(scores), dtype=torch.bool).to(self.device), -1e10)
            self.print_fn('\tstep %d: pos mean scores %.2f, hard neg mean scores i2t %.2f, t2i %.2f' % (
                step, torch.mean(torch.diag(scores)), torch.mean(torch.max(neg_scores, 1)[0]),
                torch.mean(torch.max(neg_scores, 0)[0])))
            self.print_fn(
                        '\tstep %d: high_align_loss %.4f, low_align_loss %.4f, global_align_loss %.4f, infomax_text %.4f, infomax_vid %.4f, high_cluster_loss %.4f, low_cluster_loss %.4f, global_cluster_loss %.4f, loss %.4f' % (
                    step, high_align_loss.data.item(), low_align_loss.data.item(),
                    glo_align_loss.data.item(), infomax_loss_text.data.item(),
                    infomax_loss_vid.data.item(), high_cluster_loss.data.item(),
                    low_cluster_loss.data.item(), glo_cluster_loss.data.item(), loss.data.item()))
        return loss

    def evaluate_scores(self, tst_reader):

        vid_names, cap_names = [], []
        vid_emb = []
        par_emb = []
        for tst_data in tst_reader:
            vid_names.extend(tst_data['names'])
            cap_names.extend(tst_data['text'])
            vid_enc_outs = self.forward_video_embed(tst_data)
            cap_enc_outs = self.forward_text_embed(tst_data)
            vid_emb.extend(vid_enc_outs['video_embeds'])
            par_emb.extend(cap_enc_outs['para_embeds'])
        vid_emb = torch.stack(vid_emb, 0)
        par_emb = torch.stack(par_emb, 0)

        scores = self.generate_scores(vid_emb, par_emb).cpu().numpy()
        return vid_names, cap_names, scores

    def evaluate(self, tst_reader, return_outs=False):
        vid_names, cap_names, scores = self.evaluate_scores(tst_reader)
        i2t_gts = []
        for vid_name in vid_names:
            i2t_gts.append([])
            for i, cap_name in enumerate(cap_names):
                if cap_name in tst_reader.dataset.ref_captions[vid_name]:
                    i2t_gts[-1].append(i)

        t2i_gts = {}
        for i, t_gts in enumerate(i2t_gts):
            for t_gt in t_gts:
                t2i_gts.setdefault(t_gt, [])
                t2i_gts[t_gt].append(i)

        metrics = self.calculate_metrics(scores, i2t_gts, t2i_gts)
        if return_outs:
            outs = {
                'vid_names': vid_names,
                'cap_names': cap_names,
                'scores': scores,
            }
            return metrics, outs
        else:
            return metrics

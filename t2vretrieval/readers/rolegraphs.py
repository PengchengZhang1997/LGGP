import os
import json
import numpy as np
import time
import t2vretrieval.readers.mpdata

ROLES = ['V', 'ARG1', 'ARG0', 'ARG2', 'ARG3', 'ARG4',
 'ARGM-LOC', 'ARGM-MNR', 'ARGM-TMP', 'ARGM-DIR', 'ARGM-ADV', 
 'ARGM-PRP', 'ARGM-PRD', 'ARGM-COM', 'ARGM-MOD', 'NOUN']

class RoleGraphDataset(t2vretrieval.readers.mpdata.MPDataset):
  def __init__(self, name_file, attn_ft_files, word2int_file,
    max_words_in_sent, num_verbs, num_nouns, ref_caption_file, ref_graph_file, 
    max_attn_len=20, load_video_first=False, is_train=False, _logger=None):
    if _logger is None:
      self.print_fn = print
    else:
      self.print_fn = _logger.info

    self.max_words_in_sent = max_words_in_sent
    self.is_train = is_train
    self.attn_ft_files = attn_ft_files
    self.max_attn_len = max_attn_len
    self.ref_caption_file = ref_caption_file
    self.load_video_first = load_video_first
    self.all2split = json.load(open(os.path.join(ref_caption_file, 'all2split.json')))
    self.names = np.load(name_file)
    self.word2int = json.load(open(word2int_file))
    self.seg_num = json.load(open(os.path.join(attn_ft_files, 'seg_num.json')))
    self.sent_num = json.load(open(os.path.join(ref_caption_file, 'sent_num.json')))
    self.para_len = json.load(open(os.path.join(ref_caption_file, 'para_len.json')))
    self.text_len = json.load(open(os.path.join(ref_caption_file, 'text_len.json')))
    self.video_lens = json.load(open(os.path.join(attn_ft_files, 'vid_len.json')))
    self.clips_lens = json.load(open(os.path.join(attn_ft_files, 'clip_len.json')))
    self.num_videos = len(self.names)
    self.print_fn('num_videos %d' % (self.num_videos))
    self.vid_fts = {}
    self.clip_fts = {}

    if ref_caption_file is None:
      self.ref_captions = None
    else:
      self.ref_captions = json.load(open(os.path.join(ref_caption_file, 'ref_paragraph_captions.json')))
      self.find_name = {}
      for key in self.ref_captions.keys():
        self.find_name[self.ref_captions[key][0]] = key
      self.captions = set()
      self.pair_idxs = []
      for i, name in enumerate(self.names):
        for j, sent in enumerate(self.ref_captions[name]):
          self.captions.add(sent)
          self.pair_idxs.append((i, j))
      self.captions = list(self.captions)
      self.num_pairs = len(self.pair_idxs)
      self.print_fn('captions size %d' % self.num_pairs)

    self.num_verbs = num_verbs
    self.num_nouns = num_nouns
    
    self.role2int = {}
    for i, role in enumerate(ROLES):
      self.role2int[role] = i
      self.role2int['C-%s'%role] = i
      self.role2int['R-%s'%role] = i

    self.ref_graphs = json.load(open(ref_graph_file))

  def get_caption_outs(self, name, out, para):
    #paragraph
    out['para_embs'] = np.load(os.path.join(self.ref_caption_file, 'para_new', name+'.npy'))
    out['para_lens'] = self.para_len[name]
    out['text'] = para

    #sentences
    para2sent = self.all2split
    sent_lens_all = []
    verb_mask_all = []
    noun_mask_all = []
    node_role_all = []
    rel_edges_all = []

    num = 0
    for sent in para2sent[para]:
      num += 1
      graph = self.ref_graphs[sent]
      graph_nodes, graph_edges = graph
      # print(graph_nodes, graph_edges)
      verb_node2idxs, noun_node2idxs = {}, {}
      edges = []
      node_roles = np.zeros((self.num_verbs + self.num_nouns,), np.int32)
      # root node
      sent_len = self.text_len[name][num - 1]
      # print(sent_ids, sent_len)

      # graph: add verb nodes
      node_idx = 1
      verb_masks = np.zeros((self.num_verbs, self.max_words_in_sent), np.bool)
      for knode, vnode in graph_nodes.items():
        k = node_idx - 1
        if k >= self.num_verbs:
          break
        if vnode['role'] == 'V' and np.min(vnode['spans']) < self.max_words_in_sent:
          verb_node2idxs[knode] = node_idx
          for widx in vnode['spans']:
            if widx < self.max_words_in_sent:
              verb_masks[k][widx] = True
          node_roles[node_idx - 1] = self.role2int['V']
          # add root to verb edge
          edges.append((0, node_idx))
          node_idx += 1

      # graph: add noun nodes
      node_idx = 1 + self.num_verbs
      noun_masks = np.zeros((self.num_nouns, self.max_words_in_sent), np.bool)
      for knode, vnode in graph_nodes.items():
        k = node_idx - self.num_verbs - 1
        if k >= self.num_nouns:
          break
        if vnode['role'] not in ['ROOT', 'V'] and np.min(vnode['spans']) < self.max_words_in_sent:
          noun_node2idxs[knode] = node_idx
          for widx in vnode['spans']:
            if widx < self.max_words_in_sent:
              noun_masks[k][widx] = True
          node_roles[node_idx - 1] = self.role2int.get(vnode['role'], self.role2int['NOUN'])
          node_idx += 1

      # graph: add verb_node to noun_node edges
      for e in graph_edges:
        if e[0] in verb_node2idxs and e[1] in noun_node2idxs:
          edges.append((verb_node2idxs[e[0]], noun_node2idxs[e[1]]))
          edges.append((noun_node2idxs[e[1]], verb_node2idxs[e[0]]))

      num_nodes = 1 + self.num_verbs + self.num_nouns
      rel_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
      for src_nodeidx, tgt_nodeidx in edges:
        rel_matrix[tgt_nodeidx, src_nodeidx] = 1
      # row norm
      for i in range(num_nodes):
        s = np.sum(rel_matrix[i])
        if s > 0:
          rel_matrix[i] /= s

      sent_lens_all.append(sent_len)
      verb_mask_all.append(verb_masks)
      noun_mask_all.append(noun_masks)
      node_role_all.append(node_roles)
      rel_edges_all.append(rel_matrix)

    sent_nums = self.sent_num[name]
    num_nodes = 1 + 3 * 27
    gedge = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for j in range(sent_nums):
      gedge[0][j + 1] = 1 / sent_nums
      gedge[j + 1][0] = 1
      gedge[j + 1][j + 1 + sent_nums] = 0.5
      gedge[j + 1 + sent_nums][j + 1] = 0.5
      gedge[j + 1][j + 1 + 2*sent_nums] = 0.5
      gedge[j + 1 + 2*sent_nums][j + 1] = 0.5
      gedge[j + 1 + sent_nums][j + 1 + 2*sent_nums] = 0.5
      gedge[j + 1 + 2*sent_nums][j + 1 + sent_nums] = 0.5

    out['sent_embs'] = np.load(os.path.join(self.ref_caption_file, 'sent_new', name+'.npy'))
    out['sent_lens'] = sent_lens_all
    out['verb_masks'] = verb_mask_all
    out['noun_masks'] = noun_mask_all
    out['node_roles'] = node_role_all
    out['rel_edges'] = rel_edges_all
    out['sent_nums'] = sent_nums
    out['gedges'] = gedge

    return out

  def vid_graph_edges(self, out, clip_lens):
    vid_edges = []
    for i in range(len(clip_lens)):
      num_nodes = 1 + 18 + 20
      edge = np.zeros((num_nodes, num_nodes), dtype=np.float32)
      for j in range(clip_lens[i]-2):
        edge[0][j + 1] = 1
        for k in range(clip_lens[i]):
          edge[j + 1][k + 18] = 1
          edge[k + 18][j + 1] = 1
      edge /= clip_lens[i]
      for j in range(clip_lens[i]):
        edge[j + 1][0] = 1
      vid_edges.append(edge)
    out['vid_edges'] = vid_edges

    return out

  def __getitem__(self, idx):
    out = {}
    video_idx, cap_idx = self.pair_idxs[idx]
    name = self.names[video_idx]
    para = self.ref_captions[name][cap_idx]
    out = self.get_caption_outs(name, out, para)
    vid_fts = np.load(os.path.join(self.attn_ft_files, 'vid_new', name + '.npy'))
    clip_fts = np.load(os.path.join(self.attn_ft_files, 'clip_new', name + '.npy'))
    vid_lens = self.video_lens[name]
    clip_lens = self.clips_lens[name]
    seg_num = self.seg_num[name]
    
    out['names'] = name
    out['vid_fts'] = vid_fts
    out['clip_fts'] = clip_fts
    out['vid_lens'] = vid_lens
    out['clip_lens'] = clip_lens
    out['seg_nums'] = seg_num
    out = self.vid_graph_edges(out, clip_lens)

    return out

def collate_graph_fn(data):
  outs = {}
  for key in ['names', 'vid_fts', 'vid_lens', 'clip_fts', 'clip_lens', 'seg_nums', 'vid_edges', 'para_embs', 'para_lens',
              'text', 'sent_embs', 'sent_lens', 'verb_masks', 'noun_masks', 'node_roles', 'rel_edges', 'sent_nums', 'gedges']:
    if key in data[0]:
      outs[key] = [x[key] for x in data]

  #batch_size = len(data)

  # reduce attn_lens
  max_len = np.max(outs['vid_lens'])
  outs['vid_fts'] = np.stack(outs['vid_fts'], 0)[:, :max_len]
  outs['clip_lens'] = np.concatenate(outs['clip_lens'])
  outs['clip_fts'] = np.concatenate(outs['clip_fts'])
  max_cap_len = np.max(outs['para_lens'])
  outs['para_embs'] = np.array(outs['para_embs'])[:, :max_cap_len]
  outs['sent_lens'] = np.concatenate(outs['sent_lens'])
  max_sent_len = np.max(outs['sent_lens'])
  outs['sent_embs'] = np.concatenate(outs['sent_embs'])[:, :max_sent_len]
  outs['verb_masks'] = np.concatenate(outs['verb_masks'])
  outs['noun_masks'] = np.concatenate(outs['noun_masks'])
  outs['node_roles'] = np.concatenate(outs['node_roles'])
  outs['rel_edges'] = np.concatenate(outs['rel_edges'])
  outs['vid_edges'] = np.concatenate(outs['vid_edges'])

  return outs

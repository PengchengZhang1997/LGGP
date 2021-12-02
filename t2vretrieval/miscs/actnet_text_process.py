import h5py
import json
import numpy as np

output_dir_para = '/home/zhangpengcheng/data/MM21/data/activitynet/annotation/para_new/'
output_dir_sent = '/home/zhangpengcheng/data/MM21/data/activitynet/annotation/sent_new/'
output_dir = '/home/zhangpengcheng/data/MM21/data/activitynet/annotation/'

f = h5py.File('/home/zhangpengcheng/data/MM21/data/activitynet/annotation/text_bert_embedding.h5', 'r')
lens = json.load(open('/home/zhangpengcheng/data/MM21/data/activitynet/annotation/text_para_lens.json'))

sent_nums = {}
para_lens = {}
text_lens = {}

for k in f.keys():
    para_embedding = f[k][:]
    para_lens[k] = para_embedding.shape[0]
    sent_lens = lens[k]
    sent_nums[k] = len(sent_lens)
    sent_lens.insert(0, 0)

    #sentence embedding
    text_lens[k] = []
    sent_embedding = []
    index = 0
    for i in range(len(sent_lens)-1):
        one_sent_emb = para_embedding[sent_lens[i]: index+sent_lens[i+1]]
        index += sent_lens[i+1]

        # trim for sentence
        if one_sent_emb.shape[0] >= 30:
            new_fea = one_sent_emb[:30]
            text_lens[k].append(30)

        # pad for sentence
        else:
            new_fea = np.zeros((30, 1536), np.float32)
            new_fea[:one_sent_emb.shape[0]] = one_sent_emb
            text_lens[k].append(one_sent_emb.shape[0])
        sent_embedding.append(new_fea.tolist())

    sent_embedding = np.array(sent_embedding)
    np.save(output_dir_sent + k + '.npy', sent_embedding)

    # trim for paragraph
    if para_embedding.shape[0] >= 120:
        new_fea = para_embedding[:120]
        para_lens[k] = 120

    # pad for paragraph
    else:
        new_fea = np.zeros((120, 1536), np.float32)
        new_fea[:para_embedding.shape[0]] = para_embedding
        para_lens[k] = para_embedding.shape[0]

    np.save(output_dir_para + k + '.npy', new_fea)

with open(output_dir+'para_len.json', 'w') as f:
    json.dump(para_lens, f)

with open(output_dir+'text_len.json', 'w') as f:
    json.dump(text_lens, f)

with open(output_dir+'sent_num.json', 'w') as f:
    json.dump(sent_nums, f)
import os
import json

from allennlp.predictors.predictor import Predictor

def main():
  predictor = Predictor.from_path("/home1/zhangpengcheng/data/MM21/tools/bert-base-srl-2020.03.24.tar.gz",cuda_device=0)
  #predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.03.24.tar.gz",cuda_device=0)
  #predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/bert-base-srl-2019.06.17.tar.gz", cuda_device=0)
  ref_caption_file = '/home1/zhangpengcheng/data/MM21/data/activitynet/annotation/ref_captions.json'
  out_file = '/home1/zhangpengcheng/data/MM21/data/activitynet/annotation/sent2srl.json'
  ref_caps = json.load(open(ref_caption_file))
  uniq_sents = set()
  for key, sents in ref_caps.items():
    for sent in sents:
      uniq_sents.add(sent)
  uniq_sents = list(uniq_sents)
  print('unique sents', len(uniq_sents))

  outs = {}
  if os.path.exists(out_file):
    outs = json.load(open(out_file))
  for i, sent in enumerate(uniq_sents):
    if sent in outs:
      continue
    try:
      out = predictor.predict_tokenized(sent.split())
    except KeyboardInterrupt:
      break
    except:
      continue
    outs[sent] = out
    if i % 1000 == 0:
      print('finish %d / %d = %.2f%%' % (i, len(uniq_sents), i / len(uniq_sents) * 100))

  with open(out_file, 'w') as f:
    json.dump(outs, f)

if __name__ == '__main__':
  main()
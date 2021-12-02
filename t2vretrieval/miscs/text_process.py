import json

dir = '/home1/zhangpengcheng/data/MM21/data/activitynet/annotation/'
#dir = '/home1/zhangpengcheng/data/MM21/data/youcook2/'

caption_org = json.load(open(dir+'ref_captions_paragraph.json'))
caption = {}

for k in caption_org.keys():
    for i in range(len(caption_org[k])):
        caption[k+'_'+str(i+1)] = []
        caption[k + '_' + str(i + 1)].append(caption_org[k][i])

with open(dir+'ref_captions.json', 'w') as f:
    f.write(json.dumps(caption, ensure_ascii=False, indent=1))
    #json.dump(caption, f)
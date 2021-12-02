import numpy as np
import json

def expand_video_segment(num_frames_video, start_frame_seg, stop_frame_seg, min_frames_seg = 10):
    num_frames_seg = stop_frame_seg - start_frame_seg
    changes = False
    if min_frames_seg > num_frames_video:
        min_frames_seg = num_frames_video
    if num_frames_seg < min_frames_seg:
        while True:
            if start_frame_seg > 0:
                start_frame_seg -= 1
                num_frames_seg += 1
                changes = True
            if num_frames_seg == min_frames_seg:
                break
            if stop_frame_seg < num_frames_video:
                stop_frame_seg += 1
                num_frames_seg += 1
                changes = True
            if num_frames_seg == min_frames_seg:
                break
    return start_frame_seg, stop_frame_seg, changes

input_dir = '/home/zhangpengcheng/data/MM21/data/activitynet/features/ICEP_fea/'
output_dir_clip = '/home/zhangpengcheng/data/MM21/data/activitynet/features/clip_new/'
output_dir_vid = '/home/zhangpengcheng/data/MM21/data/activitynet/features/vid_new/'
len_dir = '/home/zhangpengcheng/data/MM21/data/activitynet/features/'

meta_dict = json.load(open('/home/zhangpengcheng/data/MM21/data/activitynet/annotation/meta_all.json'))

vid_len = {}
clip_len = {}
seg_num = {}
expand = 0

for k in meta_dict.keys():
    vid_fea = np.load(input_dir+'v_'+meta_dict[k]['data_key']+'.npz')['frame_scores']
    vid_fea = vid_fea.reshape(vid_fea.shape[0], 2048)
    fps = vid_fea.shape[0] / meta_dict[k]['duration_sec']

    #trim for vid
    if vid_fea.shape[0]>=80:
        idxs = np.round(np.linspace(0, vid_fea.shape[0]-1, 80)).astype(np.int32)
        new_fea = vid_fea[idxs]
        vid_len[k] = 80

    #pad for vid
    else:
        new_fea = np.zeros((80, 2048), np.float32)
        new_fea[:vid_fea.shape[0]] = vid_fea
        vid_len[k] = vid_fea.shape[0]

    np.save(output_dir_vid + k + '.npy', new_fea)

    count = 0

    clip_len[k] = []

    seg_fea_new = []
    for seg in meta_dict[k]['segments']:
        count = count + 1
        clip_output = {}
        if seg['start_sec']>seg['stop_sec']:
            sta_time = seg['stop_sec']
            sto_time = seg['start_sec']
        else:
            sta_time = seg['start_sec']
            sto_time = seg['stop_sec']
        start_idx = int(np.floor(sta_time * fps))
        stop_idx = int(np.ceil(sto_time * fps)) + 1
        if stop_idx>vid_fea.shape[0]:
            stop_idx = vid_fea.shape[0]
        start_idx, stop_idx, change = expand_video_segment(vid_fea.shape[0], start_idx, stop_idx, min_frames_seg=10)
        if change:
            expand +=1
        clip_fea = vid_fea[start_idx: stop_idx]

        # trim for clip
        if clip_fea.shape[0] >= 20:
            idxs = np.round(np.linspace(0, clip_fea.shape[0] - 1, 20)).astype(np.int32)
            new_ft = vid_fea[idxs]
            clip_len[k].append(20)

        # pad for clip
        else:
            new_ft = np.zeros((20, 2048), np.float32)
            new_ft[:clip_fea.shape[0]] = clip_fea
            clip_len[k].append(clip_fea.shape[0])

        seg_fea_new.append(new_ft.tolist())

    seg_num[k] = count
    seg_fea_new = np.array(seg_fea_new)
    np.save(output_dir_clip+ k + '.npy', seg_fea_new)

with open(len_dir+'vid_len.json', 'w') as f:
    json.dump(vid_len, f)

with open(len_dir+'clip_len.json', 'w') as f:
    json.dump(clip_len, f)

with open(len_dir+'seg_num.json', 'w') as f:
    json.dump(seg_num, f)

print(expand)
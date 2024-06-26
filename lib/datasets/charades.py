from __future__ import division
import os
import json
from collections import OrderedDict
import numpy as np
import h5py
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
import torchtext
import pickle
from math import ceil, floor

from lib.datasets import average_to_fixed_length
from lib.core.config import config


class Charades(data.Dataset):
    # glove 300d 400001词 最后为全0<unk>
    vocab = torchtext.vocab.pretrained_aliases["glove.6B.300d"](cache='../.vector_cache')
    vocab.itos.extend(['<unk>'])
    vocab.stoi['<unk>'] = vocab.vectors.shape[0]
    vocab.vectors = torch.cat([vocab.vectors, torch.zeros(1, vocab.dim)], dim=0)
    word_embedding = nn.Embedding.from_pretrained(vocab.vectors)

    def __init__(self, split):
        """
        self.annotations:
            list(dict)
            dict内容：video(str), duration(时长，以s为单位，float) times(起止时间，以s为单位，float，shape[2]) description(str，单句无标点)
        :param split: train/val/test
        """
        super(Charades, self).__init__()

        self.vis_input_type = config.DATASET.VIS_INPUT_TYPE
        self.data_dir = config.DATA_DIR
        self.split = split

        # tacos stoi 1336个词 str->int 'PAD':0 'UNK':1335
        with open(os.path.join(self.data_dir, 'words_vocab_charades_sta.json'), 'r') as f:
            self.itos = json.load(f)['words']
        self.stoi = OrderedDict()
        for i, w in enumerate(self.itos):
            self.stoi[w] = i

        with open(os.path.join(self.data_dir, 'charades_sta_anno_pairs_{}.pkl'.format(split)), 'rb') as f:
            self.annotations = pickle.load(f)
        print(f'name:TACoS split:{split}')

    def __getitem__(self, index):
        video_id = self.annotations[index]['video']
        gt_s_time, gt_e_time = self.annotations[index]['times']
        sentence = self.annotations[index]['description']
        duration = self.annotations[index]['duration']

        # word_label: 使用tacos字典(1336) str->int
        # word_mask: 以15%概率mask 1.-mask 0.-unmask(保证不全mask/unmask)
        word_label = [self.stoi.get(w.lower(), 1335) for w in sentence.split()]
        range_i = range(len(word_label))
        if self.split in ['test', 'val']:
            word_mask = [1. if np.random.uniform(0, 1) < 0.00 else 0. for _ in range_i]
        else:
            word_mask = [1. if np.random.uniform(0, 1) < 0.15 else 0. for _ in range_i]
        if np.sum(word_mask) == 0.:
            mask_i = np.random.choice(range_i)
            word_mask[mask_i] = 1.
        if np.sum(word_mask) == len(word_mask):
            unmask_i = np.random.choice(range_i)
            word_mask[unmask_i] = 0.
        word_label = torch.tensor(word_label, dtype=torch.long)
        word_mask = torch.tensor(word_mask, dtype=torch.float)

        # word_vectors: glove词向量300d
        word_idxs = torch.tensor([self.vocab.stoi.get(w.lower(), 400000) for w in sentence.split()], dtype=torch.long)
        word_vectors = self.word_embedding(word_idxs)

        # 视觉特征[?, vis_fea_len] 视觉mask ones[?, 1]
        visual_input, visual_mask = self.get_video_features(video_id)

        # 将视觉特征均值池化成[NUM_SAMPLE_CLIPS, vis_fea_len]
        visual_input = average_to_fixed_length(visual_input)

        # 2d时间图边长num_clips=NUM_SAMPLE_CLIPS//TARGET_STRIDE
        # num_clips个new_clip
        num_clips = config.DATASET.NUM_SAMPLE_CLIPS // config.DATASET.TARGET_STRIDE

        # 2d时间图GT表 shape[5, num_clips+1] 时间单位为x个new_clip，即时间范围在[0, num_clips] numpy float32
        # num_clips个new_clip将视频分隔成num_clips+1个时间点(时间单位为x个new_clip)
        # 仅对部分满足要求的候选片段做边框回归，要求候选片段s \in [gt_s - scale*gt_l, gt_s + scale*gt_l] 且 e \in [gt_e - scale*gt_l, gt_e + scale*gt_l]
        # 0: num_clips+1个时间点是否为符合回归要求的候选时刻开始时间
        # 1: num_clips+1个时间点是否为符合回归要求的候选时刻结束时间
        # 2: 句子开始时间 与 num_clips+1个时间点 的差(时间单位为x个new_clip)
        # 3: 句子结束时间 与 num_clips+1个时间点 的差(时间单位为x个new_clip)
        # 4: 波谷式前背景监督（相对应0-1式）
        map_gt = np.zeros((5, num_clips + 1), dtype=np.float32)
        clip_duration = duration / num_clips
        gt_s = gt_s_time / clip_duration
        gt_e = gt_e_time / clip_duration
        gt_length = gt_e - gt_s
        gt_centre = (gt_s + gt_e) / 2
        map_gt[0, :] = np.exp(-0.5 * np.square((np.arange(num_clips + 1) - gt_s) / (0.25 * gt_length)))
        map_gt[0, map_gt[0, :] >= 0.7] = 1.
        map_gt[0, map_gt[0, :] < 0.1353] = 0.
        map_gt[1, :] = np.exp(-0.5 * np.square((np.arange(num_clips + 1) - gt_e) / (0.25 * gt_length)))
        map_gt[1, map_gt[1, :] >= 0.7] = 1.
        map_gt[1, map_gt[1, :] < 0.1353] = 0.
        map_gt[2, :] = gt_s - np.arange(num_clips + 1)
        map_gt[3, :] = gt_e - np.arange(num_clips + 1)
        map_gt[4, floor(gt_s):ceil(gt_e) + 1] = np.minimum(2 / np.square(gt_e - gt_s) * np.square(np.arange(num_clips + 1) - gt_centre) + 0.5, 1)[floor(gt_s):ceil(gt_e) + 1]

        if (map_gt[0, :] > 0.4).sum() == 0:
            p = np.exp(-0.5 * np.square((np.arange(num_clips + 1) - gt_s) / (0.25 * gt_length)))
            idx = np.argsort(p)
            map_gt[0, idx[-1]] = 1.
        if (map_gt[1, :] > 0.4).sum() == 0:
            p = np.exp(-0.5 * np.square((np.arange(num_clips + 1) - gt_e) / (0.25 * gt_length)))
            idx = np.argsort(p)
            map_gt[1, idx[-1]] = 1.

        item = {
            # 某 视频(完整)-文本(单个) 对的:
            # 视觉特征 shape[num_sample_clips, vis_fea_len] tensor float32
            'visual_input': visual_input,
            # 视觉特征掩码 shape[num_sample_clips, 1] ones tensor float32
            'vis_mask': visual_mask,
            # 样本idx int
            'anno_idx': index,
            # 单词特征 shape[len_sentence, text_fea_len] tensor float32
            'word_vectors': word_vectors,
            # 视频时长 以s为单位 float
            'duration': duration,
            # 文本掩码 shape[len_sentence] ones tensor float32
            'txt_mask': torch.ones(word_vectors.shape[0], ),
            'map_gt': torch.from_numpy(map_gt),
            # 单词标签(使用tacos专用1514字典) shape[len_sentence] tensor int64
            'word_label': word_label,
            # 单词掩码 shape[len_sentence] tensor float32 1.-mask 0.unmask 15%的mask率
            'word_mask': word_mask,
            # 句子起止时间 shape[2] tensor float32 时间单位为x个new_clip，即时间范围在[0, num_clips]
            'gt_times': torch.from_numpy(np.array([gt_s, gt_e], dtype=np.float32))
        }

        return item

    def __len__(self):
        return len(self.annotations)

    def get_video_features(self, vid):
        """
        获取视频vid的所有win64_overlap0.8特征[?, vis_fea_len]tensor 及 mask ones[?, 1]tensor
        可选对读入的数据进行normalize
        :param vid: 视频名
        :return:
        """
        assert config.DATASET.VIS_INPUT_TYPE == 'vgg_rgb'
        with h5py.File(os.path.join(self.data_dir, 'vgg_rgb_features.hdf5'), 'r') as f:
            features = torch.from_numpy(f[vid][:])
        if config.DATASET.NORMALIZE:
            features = F.normalize(features, dim=1)
        vis_mask = torch.ones((features.shape[0], 1))
        return features, vis_mask

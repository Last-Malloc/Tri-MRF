import torch
import torch.nn as nn

from lib.core.config import config


def collate_fn(batch):
    """

    :param batch: shape[b] 即b个item item内容见tacos.py或activitynet.py
    :return:
    """
    batch_word_vectors = [b['word_vectors'] for b in batch]
    batch_txt_mask = [b['txt_mask'] for b in batch]
    batch_map_gt = [b['map_gt'] for b in batch]
    batch_anno_idxs = [b['anno_idx'] for b in batch]
    batch_vis_feats = [b['visual_input'] for b in batch]
    batch_duration = [b['duration'] for b in batch]
    batch_word_label = [b['word_label'] for b in batch]
    batch_word_mask = [b['word_mask'] for b in batch]
    batch_gt_times = [b['gt_times'].unsqueeze(0) for b in batch]

    max_num_clips = max([map_gt.shape[-1] for map_gt in batch_map_gt])
    padded_batch_map_gt = torch.zeros(len(batch_map_gt), 5, max_num_clips)
    for i, map_gt in enumerate(batch_map_gt):
        num_clips = map_gt.shape[-1]
        padded_batch_map_gt[i][:, :num_clips] = map_gt

    batch_data = {
        # 样本idx shape[b] int
        'batch_anno_idxs': batch_anno_idxs,
        # 单词特征 shape[b, batch_max_len_sentence, text_fea_len] tensor float32 经0-pad_sequence
        'batch_word_vectors': nn.utils.rnn.pad_sequence(batch_word_vectors, batch_first=True),
        # 文本掩码 shape[b, batch_max_len_sentence] ones||zeros tensor float32 经0-pad_sequence
        'batch_txt_mask': nn.utils.rnn.pad_sequence(batch_txt_mask, batch_first=True),
        # 2d时间图GT表 shape[5, num_clips+1] 时间单位为x个new_clip，即时间范围在[0, num_clips] numpy float32
        # num_clips个new_clip将视频分隔成num_clips+1个时间点(时间单位为x个new_clip)
        # 仅对部分满足要求的候选片段做边框回归，要求候选片段s \in [gt_s - scale*gt_l, gt_s + scale*gt_l] 且 e \in [gt_e - scale*gt_l, gt_e + scale*gt_l]
        # ，其中gt_l = gt_e - gt_s，暂取scale=sqrt(-ln(0.4)/8)=0.33843218151391774
        # 0: num_clips+1个时间点是否为符合回归要求的候选时刻开始时间
        # 1: num_clips+1个时间点是否为符合回归要求的候选时刻结束时间
        # 2: 句子开始时间 与 num_clips+1个时间点 的差(时间单位为x个new_clip)
        # 3: 句子结束时间 与 num_clips+1个时间点 的差(时间单位为x个new_clip)
        # 4: 谷式前背景监督（相对应0-1式）
        'batch_map_gt': padded_batch_map_gt,
        # 视觉特征 shape[b, num_sampel_clips, vis_fea_len] tensor float32
        'batch_vis_input': nn.utils.rnn.pad_sequence(batch_vis_feats, batch_first=True).float(),
        # 视频时长 以s为单位 shape[b] float
        'batch_duration': batch_duration,
        # 单词标签(使用tacos专用1514字典) shape[b, batch_max_len_sentence] tensor int64 经0-pad_sequence
        'batch_word_label': nn.utils.rnn.pad_sequence(batch_word_label, batch_first=True).long(),
        # 单词掩码 shape[[b, batch_max_len_sentence] tensor float32 1.-mask 0.unmask 15%的mask率 经0-pad_sequence
        'batch_word_mask': nn.utils.rnn.pad_sequence(batch_word_mask, batch_first=True).float(),
        # 句子起止时间 shape[b, 2] tensor float32 时间单位为x个new_clip，即时间范围在[0, num_clips]
        'batch_gt_times': torch.cat(batch_gt_times, 0)
    }

    return batch_data


def average_to_fixed_length(visual_input):
    """
    将某视频全部视觉特征 通过均值池化为 NUM_SAMPLE_CLIPS个视觉特征
    :param visual_input: [?, vis_fea_len]
    :return:
    """
    num_sample_clips = config.DATASET.NUM_SAMPLE_CLIPS
    num_clips = visual_input.shape[0]
    idxs = torch.arange(0, num_sample_clips+1, 1.0)/num_sample_clips*num_clips
    idxs = torch.min(torch.round(idxs).long(), torch.tensor(num_clips-1))
    new_visual_input = []
    for i in range(num_sample_clips):
        s_idx, e_idx = idxs[i].item(), idxs[i+1].item()
        if s_idx < e_idx:
            new_visual_input.append(torch.mean(visual_input[s_idx: e_idx], dim=0))
        else:
            new_visual_input.append(visual_input[s_idx])
    new_visual_input = torch.stack(new_visual_input, dim=0)
    return new_visual_input


from lib.datasets.activitynet import ActivityNet
from lib.datasets.tacos import TACoS
from lib.datasets.charades import Charades

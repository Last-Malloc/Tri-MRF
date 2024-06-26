import torch
from torch.nn import functional as F


def bce_rescale_loss(config, logits_text, logits_iou, iou_mask_map, gt_maps, gt_times, word_label, word_mask, logits_backfore):
    """

    :param config: loss config params
    :param logits_text: [b, batch_max_len_sentence, vocab_size] predict mask word(包含pad出单词的预测，但后面仅对遮盖单词计算损失)
    :param logits_iou: [b, 3, num_clips + 1, num_clips + 1] 任意开始-任意终止时间点的 回归分数+对齐分数
    :param iou_mask_map: [num_clips + 1, num_clips + 1] 2D时间图mask
    :param gt_maps:
        # 2d时间图GT表 shape[b, 5, num_clips+1] 时间单位为x个new_clip，即时间范围在[0, num_clips] numpy float32
        # num_clips个new_clip将视频分隔成num_clips+1个时间点(时间单位为x个new_clip)
        # 仅对部分满足要求的候选片段做边框回归，要求候选片段s \in [gt_s - scale*gt_l, gt_s + scale*gt_l] 且 e \in [gt_e - scale*gt_l, gt_e + scale*gt_l]
        # ，其中gt_l = gt_e - gt_s，暂取scale=sqrt(-ln(0.4)/8)=0.33843218151391774
        # 0: num_clips+1个时间点是否为符合回归要求的候选时刻开始时间
        # 1: num_clips+1个时间点是否为符合回归要求的候选时刻结束时间
        # 2: 句子开始时间 与 num_clips+1个时间点 的差(时间单位为x个new_clip)
        # 3: 句子结束时间 与 num_clips+1个时间点 的差(时间单位为x个new_clip)
        # 4: 谷式前背景监督（相对应0-1式）
    :param gt_times: 句子起止时间 shape[b, 2] tensor float32 时间单位为x个new_clip，即时间范围在[0, num_clips]
    :param word_label: 单词标签(使用tacos专用1514字典) shape[b, batch_max_len_sentence] tensor int64
    :param word_mask: 单词掩码 shape[b, batch_max_len_sentence] tensor float32 1.-mask 0.-unmask 15%的mask率
    :param logits_backfore: 前背景概率值 [b, num_clips + 1]
    :return:
        loss_value: 最终loss scalar
        joint_prob: stage specific score(start end middle, 经过sigmoid) [b, 3, num_clips+1]
        torch.sigmoid(logits_iou[:, 2, :, :]) * temp: 对齐分数，经过sigmoid，只取开始时间点<结束时间点&iou_mask_map==1的位置(其他位置全0) [b, num_clips+1, num_clips+1]
        s_e_time: (num_clips + 1)^2个候选时刻的起止时间（经过regression调整，时间单位为x个new_clip） [b, 2, num_clips + 1, num_clips + 1]
    """
    # T = num_clips + 1，T个时间点
    T = gt_maps.shape[-1]

    # loss backfore [b]
    backfore_prob = torch.sigmoid(logits_backfore)
    gt_backfore = gt_maps[:, 4, :]
    loss = F.binary_cross_entropy_with_logits(logits_backfore, gt_backfore, reduction='none') * (backfore_prob - gt_backfore) * (backfore_prob - gt_backfore)
    loss = loss.sum(-1)

    # loss regress [b, 2]
    # [b, 1, num_clips + 1, num_clips + 1]
    reg_mask = gt_maps[:, 0:1, :T, None] * gt_maps[:, 1:2, None, :T]
    # [b, 2, num_clips + 1, num_clips + 1]
    gt_tmp = torch.cat((gt_maps[:, 2:3, :T, None].repeat(1, 1, 1, T), gt_maps[:, 3:4, None, :T].repeat(1, 1, T, 1)), 1)
    # [b, 2]
    loss_reg = (torch.abs(logits_iou[:, :2, :, :] - gt_tmp) * reg_mask).sum((2, 3)) / reg_mask.sum((2, 3))

    # loss match [b] 只计算开始时间点<结束时间点&iou_mask_map==1的位置损失
    # s_e_time (num_clips + 1)^2个候选时刻的起止时间（经过regression调整，时间单位为x个new_clip） [b, 2, num_clips + 1, num_clips + 1]
    idxs = torch.arange(T, device=logits_iou.device)
    s_e_idx = torch.cat((idxs[None, None, :T, None].repeat(1, 1, 1, T), idxs[None, None, None, :T].repeat(1, 1, T, 1)), 1)
    s_e_time = (s_e_idx + logits_iou[:, :2, :, :]).clone().detach()
    # (num_clips + 1)^2个候选时刻 与 文本描述GT 的时间IoU（若<0.5则赋值为0） [b, num_clips+1, num_clips+1]
    iou = torch.clamp(torch.min(gt_times[:, 1][:, None, None], s_e_time[:, 1, :, :]) - torch.max(gt_times[:, 0][:, None, None], s_e_time[:, 0, :, :]), min=0.0000000001) \
          / torch.clamp(torch.max(gt_times[:, 1][:, None, None], s_e_time[:, 1, :, :]) - torch.min(gt_times[:, 0][:, None, None], s_e_time[:, 0, :, :]), min=0.0000001)
    iou[iou < 0.5] = 0.
    temp = (s_e_time[:, 0, :, :] < s_e_time[:, 1, :, :]) * iou_mask_map[None, :T, :]  # [b, num_clips+1, num_clips+1]
    loss_iou = (F.binary_cross_entropy_with_logits(logits_iou[:, 2, :, :], iou, reduction='none') * temp * torch.pow(torch.sigmoid(logits_iou[:, 2, :, :]) - iou, 2)).sum((1, 2)) / temp.sum((1, 2))

    # loss masked word_prediction scalar 只计算mask位置损失
    # [b, batch_max_len_sentence, vocab_size]
    log_p = F.log_softmax(logits_text, -1) * word_mask.unsqueeze(2)
    # [b, batch_max_len_sentence, vocab_size]
    grid = torch.arange(log_p.shape[-1], device=log_p.device).repeat(log_p.shape[0], log_p.shape[1], 1)
    # salar
    text_loss = torch.sum(-log_p[grid == word_label.unsqueeze(2)]) / torch.clamp((word_mask.sum(1) > 0).sum(), min=0.00000001)

    loss_value = config.W1 * loss.mean() + config.W2 * loss_reg.mean() + config.W3 * loss_iou.mean() + config.W4 * text_loss

    return loss_value, torch.sigmoid(logits_iou[:, 2, :, :]) * temp, s_e_time

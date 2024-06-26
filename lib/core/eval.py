import numpy as np
from terminaltables import AsciiTable

from lib.core.config import config


def iou(pred, gt):
    """
    计算iou numpy float32
    :param pred: list [2] i / [p, 2] ii
    :param gt: list [2] iii / [g, 2] iv
    :return:
        i iii: scale
        i iv: [g]
        ii iii: [p]
        ii iv: [p, g]
    """
    assert isinstance(pred, list) and isinstance(gt, list)
    pred_is_list = isinstance(pred[0], list)
    gt_is_list = isinstance(gt[0], list)
    if not pred_is_list:
        pred = [pred]
    if not gt_is_list:
        gt = [gt]
    pred, gt = np.array(pred), np.array(gt)
    inter_left = np.maximum(pred[:, 0, None], gt[None, :, 0])
    inter_right = np.minimum(pred[:, 1, None], gt[None, :, 1])
    inter = np.maximum(0.0, inter_right - inter_left)
    union_left = np.minimum(pred[:, 0, None], gt[None, :, 0])
    union_right = np.maximum(pred[:, 1, None], gt[None, :, 1])
    union = np.maximum(0.0, union_right - union_left)
    overlap = 1.0 * inter / union
    if not gt_is_list:
        overlap = overlap[:, 0]
    if not pred_is_list:
        overlap = overlap[0]
    return overlap


def nms(dets, thresh=0.4, top_k=-1):
    """
    非最大抑制
    :param dets: list[?, 2]
    :param thresh: 阈值
    :param top_k: 返回元素个数
    :return: list[??, 2] ??<=min(?, top_k)
    """
    if len(dets) == 0:
        return []
    order = np.arange(0, len(dets), 1)
    dets = np.array(dets)
    x1 = dets[:, 0]
    x2 = dets[:, 1]
    lengths = x2 - x1
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if len(keep) == top_k:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1)
        ovr = inter / (lengths[i] + lengths[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return dets[keep]


def eval(segments, data):
    """

    :param segments: 最终预测起止时间，根据对齐分数从高到低排序，时间单位为s
            list[len_test_dataset * sub_list] sub_list的shape为[?, 2] ?为开始时间<终止时间的预测数量（最后可能包含部分 开始时间点<结束时间点|iou_mask_map==0 但 开始时间<终止时间 的非法样例）
    :param data: list(dict)
            dict内容：video(str), duration(时长，以s为单位，float) times(起止时间，以s为单位，float，shape[2]) description(str，单句无标点)
    :return:
        eval_result: list shape[len(tious), len(recalls)] recall@?,IoU=?
        miou: 每个测试 视频-文本 对，第1个预测结果的均值 float
    """
    tious = [float(i) for i in config.TEST.TIOU.split(',')] if isinstance(config.TEST.TIOU, str) else [config.TEST.TIOU]
    recalls = [int(i) for i in config.TEST.RECALL.split(',')] if isinstance(config.TEST.RECALL, str) else [config.TEST.RECALL]

    # list shape[len(tious), len(recalls), 0]
    eval_result = [[[] for _ in recalls] for _ in tious]
    max_recall = max(recalls)
    average_iou = []
    for seg_raw, dat in zip(segments, data):
        for i, t in enumerate(tious):
            # list shape[max_recall, 2] 非最大抑制获取排名前max_recall个预测结果
            seg = nms(seg_raw, thresh=min(t - 0.05, config.TEST.NMS_THRESH), top_k=max_recall).tolist()
            # numpy [max_recall, 1] 前max_recall个预测结果与GT的iou
            overlap = iou(seg, [dat['times']])

            # 排名第1的结果与GT的iou
            average_iou.append(np.mean(np.sort(overlap[0])[-3:]))

            for j, r in enumerate(recalls):
                # 前r个是否存在iou>t的值，有则append(True)，否则append(False) 最终eval_result->shape[len(tious), len(recalls), max_recall]
                eval_result[i][j].append((overlap > t)[:r].any())
    # list shape[len(tious), len(recalls)] recall@?,IoU=?
    eval_result = np.array(eval_result).mean(axis=-1)
    # scalar
    miou = np.mean(average_iou)

    return eval_result, miou


def eval_predictions(segments, data, verbose=True):
    """

    :param segments: 最终预测起止时间，根据对齐分数从高到低排序，时间单位为s
            list[len_test_dataset * sub_list] sub_list的shape为[?, 2] ?为开始时间<终止时间的预测数量（最后可能包含部分 开始时间点<结束时间点|iou_mask_map==0 但 开始时间<终止时间 的非法样例）
    :param data: list(dict)
            dict内容：video(str), duration(时长，以s为单位，float) times(起止时间，以s为单位，float，shape[2]) description(str，单句无标点)
    :param verbose: 是否打印结果，默认True
    :return:
        eval_result: list shape[len(tious), len(recalls)] recall@?,IoU=?
        miou: 每个测试 视频-文本 对，第1个预测结果的均值 float
    """
    eval_result, miou = eval(segments, data)
    if verbose:
        print(display_results(eval_result, miou, ''))

    return eval_result, miou


def display_results(eval_result, miou, title=None):
    """
    返回打印字符串
    :param eval_result: list shape[len(tious), len(recalls)] recall@?,IoU=?
    :param miou: 每个测试 视频-文本 对，第1个预测结果的均值 float
    :param title: 打印表格的标题
    :return: 返回打印字符串（不输出），eg:

    +performance on testing set---------+-----------------+-----------------+-----------------+-----------------+-----------------+-----------------+-----------------+-----------------+-------+
    | Rank@1,mIoU@0.1 | Rank@1,mIoU@0.3 | Rank@1,mIoU@0.5 | Rank@1,mIoU@0.7 | Rank@1,mIoU@0.9 | Rank@5,mIoU@0.1 | Rank@5,mIoU@0.3 | Rank@5,mIoU@0.5 | Rank@5,mIoU@0.7 | Rank@5,mIoU@0.9 | mIoU  |
    +-----------------+-----------------+-----------------+-----------------+-----------------+-----------------+-----------------+-----------------+-----------------+-----------------+-------+
    |      59.26      |      48.79      |      37.57      |      23.97      |       4.75      |      75.78      |      67.66      |      57.94      |      33.44      |       6.00      | 34.89 |
    +-----------------+-----------------+-----------------+-----------------+-----------------+-----------------+-----------------+-----------------+-----------------+-----------------+-------+

    """
    tious = [float(i) for i in config.TEST.TIOU.split(',')] if isinstance(config.TEST.TIOU, str) else [config.TEST.TIOU]
    recalls = [int(i) for i in config.TEST.RECALL.split(',')] if isinstance(config.TEST.RECALL, str) else [config.TEST.RECALL]

    display_data = [['Rank@{},mIoU@{}'.format(i, j) for i in recalls for j in tious] + ['mIoU']]
    eval_result = eval_result * 100
    miou = miou * 100
    display_data.append(['{:.02f}'.format(eval_result[j][i]) for i in range(len(recalls)) for j in range(len(tious))]
                        + ['{:.02f}'.format(miou)])
    table = AsciiTable(display_data, title)
    for i in range(len(tious) * len(recalls)):
        table.justify_columns[i] = 'center'
    return table.table

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import lib.models.loss as loss
from lib import datasets, models
from lib.core.config import config, update_config
from lib.core.engine import Engine
from lib.core.utils import AverageMeter
from lib.core.eval import eval_predictions, display_results

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.set_printoptions(precision=2, sci_mode=False)


def parse_args():
    parser = argparse.ArgumentParser(description='Test localization network')
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--split', default='test', type=str)
    args = parser.parse_args()
    update_config(args.cfg)
    return args


if __name__ == '__main__':

    args = parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    model = getattr(models, config.MODEL.NAME)()
    model_checkpoint = torch.load(config.MODEL.CHECKPOINT)
    model.load_state_dict(model_checkpoint, strict=False)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    model.eval()

    test_dataset = getattr(datasets, config.DATASET.NAME)(args.split)
    dataloader = DataLoader(test_dataset,
                            batch_size=config.TEST.BATCH_SIZE,
                            shuffle=False,
                            num_workers=config.WORKERS,
                            pin_memory=False,
                            collate_fn=datasets.collate_fn)

    def network(sample):
        textual_input = sample['batch_word_vectors'].cuda()
        textual_mask = sample['batch_txt_mask'].cuda()
        visual_input = sample['batch_vis_input'].cuda()
        map_gt = sample['batch_map_gt'].cuda()
        duration = sample['batch_duration']
        word_label = sample['batch_word_label'].cuda()
        word_mask = sample['batch_word_mask'].cuda()
        gt_times = sample['batch_gt_times'].cuda()

        logits_text, logits_iou, iou_mask_map, logits_backfore = model(textual_input, textual_mask, word_mask, visual_input)
        loss_value, iou_scores, regress = getattr(loss, config.LOSS.NAME)(config.LOSS.PARAMS, logits_text,
                                                                          logits_iou,
                                                                          iou_mask_map, map_gt, gt_times,
                                                                          word_label, word_mask, logits_backfore)

        sorted_times = None if model.training else get_proposal_results(iou_scores, regress, duration)

        return loss_value, sorted_times

    def get_proposal_results(scores, regress, durations):
        out_sorted_times = []
        T = scores.shape[-1]
        regress = regress.cpu().detach().numpy()
        scores = scores.cpu().detach().numpy()

        for score, reg, duration in zip(scores, regress, durations):
            # (num_clips + 1)^2个候选时刻从高到低排序（开始时间点<结束时间点|iou_mask_map==0由于对齐分数为0将跟在后面）
            sorted_indexs = np.dstack(
                np.unravel_index(np.argsort(score.ravel())[::-1], (T, T))).tolist()
            sorted_indexs = np.array(
                [[reg[0, item[0], item[1]], reg[1, item[0], item[1]]] for item in sorted_indexs[0] if
                 reg[0, item[0], item[1]] < reg[1, item[0], item[1]]]).astype(float)
            target_size = config.DATASET.NUM_SAMPLE_CLIPS // config.DATASET.TARGET_STRIDE
            out_sorted_times.append((sorted_indexs / target_size * duration).tolist())

        return out_sorted_times


    def on_test_start(state):
        state['loss_meter'] = AverageMeter()
        state['sorted_segments_list'] = []
        state['output'] = []
        if config.VERBOSE:
            state['progress_bar'] = tqdm(total=math.ceil(len(test_dataset)/config.TRAIN.BATCH_SIZE))

    def on_test_forward(state):
        if config.VERBOSE:
            state['progress_bar'].update(1)
        state['loss_meter'].update(state['loss'].item(), 1)

        min_idx = min(state['sample']['batch_anno_idxs'])
        batch_indexs = [idx - min_idx for idx in state['sample']['batch_anno_idxs']]
        sorted_segments = [state['output'][i] for i in batch_indexs]
        state['sorted_segments_list'].extend(sorted_segments)

    def on_test_end(state):
        if config.VERBOSE:
            state['progress_bar'].close()
            print()

        annotations = test_dataset.annotations
        state['Rank@N,mIoU@M'], state['miou'] = eval_predictions(state['sorted_segments_list'], annotations, verbose=True)

        loss_message = '\ntest loss {:.4f}'.format(state['loss_meter'].avg)
        print(loss_message)
        state['loss_meter'].reset()
        test_table = display_results(state['Rank@N,mIoU@M'], state['miou'],
                                          'performance on testing set')
        table_message = '\n'+test_table
        print(table_message)


    engine = Engine()
    engine.hooks['on_test_start'] = on_test_start
    engine.hooks['on_test_forward'] = on_test_forward
    engine.hooks['on_test_end'] = on_test_end
    engine.test(network, dataloader, args.split)

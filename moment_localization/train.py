# In the original release of the code, we inadvertently used the time-consuming masked_select and index_select operations inside loops. We fixed this in an updated version of vlbert_new.py by using masks to cover invalid positions in the 2D temporal map, but did not release new checkpoints for time reasons. Theoretically, the fix does not affect model performance, only FLOPs (see vlbert_new.py for details).

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import pprint
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import math

from lib import datasets, models
from lib.core.config import config, update_config
from lib.core.engine import Engine
from lib.core.utils import AverageMeter, create_logger
from lib.core import eval
import lib.models.loss as loss

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.autograd.set_detect_anomaly(True)


def parse_args():
    parser = argparse.ArgumentParser(description='Test localization network')
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    args = parser.parse_args()
    update_config(args.cfg)
    return args


if __name__ == '__main__':

    args = parse_args()

    # 创建日志文件，向其中写入当前配置config中的所有内容
    logger = create_logger(config, args.cfg)
    logger.info('\n' + pprint.pformat(config))

    # cudnn相关参数
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    # 读取数据集(test_dataset，当EVAL_TRAIN时同时读取eval_train_dataset, 当not NO_VAL时同时读取val_dataset)
    dataset_name = config.DATASET.NAME
    train_dataset = getattr(datasets, dataset_name)('train')
    if config.TEST.EVAL_TRAIN:
        eval_train_dataset = getattr(datasets, dataset_name)('train')
    if not config.DATASET.NO_VAL:
        val_dataset = getattr(datasets, dataset_name)('val')
    test_dataset = getattr(datasets, dataset_name)('test')

    # 创建模型
    # 若CONTINUE 则从已有检查点继续训练
    # 若多个设备 则并行训练
    # 若cuda可用 则使用cuda
    model_name = config.MODEL.NAME
    model = getattr(models, model_name)()
    if config.MODEL.CHECKPOINT and config.TRAIN.CONTINUE:
        model_checkpoint = torch.load(config.MODEL.CHECKPOINT)
        model.load_state_dict(model_checkpoint, state_dict=False)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model)
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 优化器 和 lr_scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config.TRAIN.LR, betas=(0.9, 0.999),
                            weight_decay=config.TRAIN.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config.TRAIN.FACTOR,
                                                     patience=config.TRAIN.PATIENCE, verbose=config.VERBOSE)


    def iterator(split):
        """
        通过dataset(class)构造dataloader(class)
        train用于训练，shuffle/no_shuffle，train.batch_size
        val/test/train_no_shuffle用于测试，全部no_shuffle，test.batch_size
        :param split: train/val/test/train_no_shuffle
        :return: dataloader
        """
        if split == 'train':
            dataloader = DataLoader(train_dataset,
                                    batch_size=config.TRAIN.BATCH_SIZE,
                                    shuffle=config.TRAIN.SHUFFLE,
                                    num_workers=config.WORKERS,
                                    pin_memory=False,
                                    collate_fn=datasets.collate_fn)
        elif split == 'val':
            dataloader = DataLoader(val_dataset,
                                    batch_size=config.TEST.BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=config.WORKERS,
                                    pin_memory=False,
                                    collate_fn=datasets.collate_fn)
        elif split == 'test':
            dataloader = DataLoader(test_dataset,
                                    batch_size=config.TEST.BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=config.WORKERS,
                                    pin_memory=False,
                                    collate_fn=datasets.collate_fn)
        elif split == 'train_no_shuffle':
            dataloader = DataLoader(eval_train_dataset,
                                    batch_size=config.TEST.BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=config.WORKERS,
                                    pin_memory=False,
                                    collate_fn=datasets.collate_fn)
        else:
            raise NotImplementedError

        return dataloader


    def network(sample):
        """
        网络模型调用
        :param sample: 一批次输入数据
        :return:
            model.training 不返回
            not model.training 返回sorted_times
        """
        # 数据从cpu->cuda
        textual_input = sample['batch_word_vectors'].cuda()
        textual_mask = sample['batch_txt_mask'].cuda()
        visual_input = sample['batch_vis_input'].cuda()
        map_gt = sample['batch_map_gt'].cuda()
        duration = sample['batch_duration']
        word_label = sample['batch_word_label'].cuda()
        word_mask = sample['batch_word_mask'].cuda()
        gt_times = sample['batch_gt_times'].cuda()

        # 输入数据到模型中，获得输出
        logits_text, logits_iou, iou_mask_map, logits_backfore = model(textual_input, textual_mask, word_mask, visual_input)
        # 根据输出和GT，获取loss
        loss_value, iou_scores, regress = getattr(loss, config.LOSS.NAME)(config.LOSS.PARAMS, logits_text,
                                                                                      logits_iou,
                                                                                      iou_mask_map, map_gt, gt_times,
                                                                                      word_label, word_mask, logits_backfore)
        # 最终预测起止时间，根据对齐分数从高到低排序，时间单位为s
        # list[b * sub_list] sub_list的shape为[?, 2] ?为开始时间<终止时间的预测数量（最后可能包含部分 开始时间点<结束时间点|iou_mask_map==0 但 开始时间<终止时间 的非法样例）
        sorted_times = None if model.training else get_proposal_results(iou_scores, regress, duration)

        return loss_value, sorted_times


    def get_proposal_results(scores, regress, durations):
        """
        :param scores: 对齐分数，经过sigmoid，只取开始时间点<结束时间点&iou_mask_map==1的位置(其他位置全0) [b, num_clips+1, num_clips+1]
        :param regress: (num_clips + 1)^2个候选时刻的起止时间（经过regression调整） [b, 2, num_clips + 1, num_clips + 1]
        :param durations: 视频时长 以s为单位 shape[b] float
        :return: 最终预测起止时间，根据对齐分数从高到低排序，时间单位为s
            list[b * sub_list] sub_list的shape为[?, 2] ?为开始时间<终止时间的预测数量（最后可能包含部分 开始时间点<结束时间点|iou_mask_map==0 但 开始时间<终止时间 的非法样例）
        """
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


    def on_start(state):
        state['loss_meter'] = AverageMeter()
        state['test_interval'] = int(len(train_dataset) / config.TRAIN.BATCH_SIZE * config.TEST.INTERVAL)
        state['t'] = 1
        model.train()
        if config.VERBOSE:
            state['progress_bar'] = tqdm(total=state['test_interval'])


    def on_forward(state):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        state['loss_meter'].update(state['loss'].item(), 1)


    def on_update(state):
        if config.VERBOSE:
            state['progress_bar'].update(1)

        if state['t'] % state['test_interval'] == 0:
            model.eval()
            if config.VERBOSE:
                state['progress_bar'].close()

            loss_message = '\niter: {} train loss {:.4f}'.format(state['t'], state['loss_meter'].avg)
            table_message = ''
            if config.TEST.EVAL_TRAIN:
                train_state = engine.test(network, iterator('train_no_shuffle'), 'train')
                train_table = eval.display_results(train_state['Rank@N,mIoU@M'], train_state['miou'],
                                                   'performance on training set')
                table_message += '\n' + train_table
            if not config.DATASET.NO_VAL:
                val_state = engine.test(network, iterator('val'), 'val')
                loss_message += ' val loss {:.4f}'.format(val_state['loss_meter'].avg)
                val_state['loss_meter'].reset()
                val_table = eval.display_results(val_state['Rank@N,mIoU@M'], val_state['miou'],
                                                 'performance on validation set')
                table_message += '\n' + val_table

            test_state = engine.test(network, iterator('test'), 'test')
            state['scheduler'].step(test_state['loss_meter'].avg)
            loss_message += ' test loss {:.4f}'.format(test_state['loss_meter'].avg)
            test_state['loss_meter'].reset()
            test_table = eval.display_results(test_state['Rank@N,mIoU@M'], test_state['miou'],
                                              'performance on testing set')
            table_message += '\n' + test_table

            message = loss_message + table_message + '\n'
            logger.info(message)

            saved_model_filename = os.path.join(config.MODEL_DIR, '{}/{}/iter{:06d}.pkl'.format(
                dataset_name, model_name + '_' + config.DATASET.VIS_INPUT_TYPE,
                state['t']))

            rootfolder1 = os.path.dirname(saved_model_filename)
            rootfolder2 = os.path.dirname(rootfolder1)
            rootfolder3 = os.path.dirname(rootfolder2)
            if not os.path.exists(rootfolder3):
                print('Make directory %s ...' % rootfolder3)
                os.mkdir(rootfolder3)
            if not os.path.exists(rootfolder2):
                print('Make directory %s ...' % rootfolder2)
                os.mkdir(rootfolder2)
            if not os.path.exists(rootfolder1):
                print('Make directory %s ...' % rootfolder1)
                os.mkdir(rootfolder1)

            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), saved_model_filename)
            else:
                torch.save(model.state_dict(), saved_model_filename)

            if config.VERBOSE:
                state['progress_bar'] = tqdm(total=state['test_interval'])
            model.train()
            state['loss_meter'].reset()


    def on_end(state):
        if config.VERBOSE:
            state['progress_bar'].close()


    def on_test_start(state):
        state['loss_meter'] = AverageMeter()
        state['sorted_segments_list'] = []
        if config.VERBOSE:
            if state['split'] == 'train':
                state['progress_bar'] = tqdm(total=math.ceil(len(train_dataset) / config.TEST.BATCH_SIZE))
            elif state['split'] == 'val':
                state['progress_bar'] = tqdm(total=math.ceil(len(val_dataset) / config.TEST.BATCH_SIZE))
            elif state['split'] == 'test':
                state['progress_bar'] = tqdm(total=math.ceil(len(test_dataset) / config.TEST.BATCH_SIZE))
            else:
                raise NotImplementedError


    def on_test_forward(state):
        if config.VERBOSE:
            state['progress_bar'].update(1)
        state['loss_meter'].update(state['loss'].item(), 1)

        min_idx = min(state['sample']['batch_anno_idxs'])
        batch_indexs = [idx - min_idx for idx in state['sample']['batch_anno_idxs']]
        sorted_segments = [state['output'][i] for i in batch_indexs]
        state['sorted_segments_list'].extend(sorted_segments)


    def on_test_end(state):
        annotations = state['iterator'].dataset.annotations
        state['Rank@N,mIoU@M'], state['miou'] = eval.eval_predictions(state['sorted_segments_list'], annotations,
                                                                      verbose=False)
        if config.VERBOSE:
            state['progress_bar'].close()


    engine = Engine()
    engine.hooks['on_start'] = on_start
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_update'] = on_update
    engine.hooks['on_end'] = on_end
    engine.hooks['on_test_start'] = on_test_start
    engine.hooks['on_test_forward'] = on_test_forward
    engine.hooks['on_test_end'] = on_test_end
    engine.train(network,
                 iterator('train'),
                 maxepoch=config.TRAIN.MAX_EPOCH,
                 optimizer=optimizer,
                 scheduler=scheduler)

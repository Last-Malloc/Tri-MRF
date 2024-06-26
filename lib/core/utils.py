from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import logging
import time
from pathlib import Path


class AverageMeter(object):
    """
    compute and store val(current val) avg sum count
    """
    def __init__(self):
        self.count = None
        self.sum = None
        self.avg = None
        self.val = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_logger(cfg, cfg_file_name, tag='train'):
    """
    创建 日志文件 及 日志对象
    目录为：cfg.LOG_DIR / cfg.DATASET.NAME / cfg_file_name(去除路径前缀并去除.yaml后缀)
    文件为：<cfg_file_name>_<time.strftime('%Y-%m-%d-%H-%M')>_<tag>.log

    注：每次写日志第1行将写入详细时间
    写入示例：logger.info('\n'+pprint.pformat(config))，'\n'是因为写入详细时间(head)后应换行提高可读性
    :param cfg: 配置对象 config
    :param cfg_file_name: 配置文件名，eg: MSAT-128.yaml
    :param tag: 日志文件名后缀
    :return: 日志对象 logger
    """
    cfg_file_name = os.path.basename(cfg_file_name).split('.yaml')[0]

    final_log_dir = Path(cfg.LOG_DIR) / cfg.DATASET.NAME / cfg_file_name
    final_log_dir.mkdir(parents=True, exist_ok=True)

    final_log_file = final_log_dir / '{}_{}_{}.log'.format(cfg_file_name, time.strftime('%Y-%m-%d-%H-%M'), tag)

    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger

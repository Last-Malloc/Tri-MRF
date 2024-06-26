from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import yaml
from easydict import EasyDict as edict

config = edict()

# general
config.WORKERS = 4
config.MODEL_DIR = './checkpoints'
config.LOG_DIR = './log'
config.DATA_DIR = ''
config.VERBOSE = True

# cudnn
config.CUDNN = edict()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True

# dataset
config.DATASET = edict()
config.DATASET.NAME = ''
config.DATASET.VIS_INPUT_TYPE = 'c3d'
config.DATASET.NO_VAL = True
config.DATASET.NUM_SAMPLE_CLIPS = 256
config.DATASET.TARGET_STRIDE = 2
config.DATASET.NORMALIZE = True

# model
config.MODEL = edict()
config.MODEL.NAME = ''
config.MODEL.CHECKPOINT = ''

# TAN
config.TAN = edict()
config.TAN.FRAME_MODULE = edict()
config.TAN.FRAME_MODULE.NAME = ''
config.TAN.FRAME_MODULE.PARAMS = None
config.TAN.VLBERT_MODULE = edict()
config.TAN.VLBERT_MODULE.NAME = ''
config.TAN.VLBERT_MODULE.PARAMS = None

# loss
config.LOSS = edict()
config.LOSS.NAME = 'bce_rescale_loss'
config.LOSS.PARAMS = None

# train
config.TRAIN = edict()
config.TRAIN.BATCH_SIZE = 16
config.TRAIN.LR = 0.0001
config.TRAIN.WEIGHT_DECAY = 0
config.TRAIN.MAX_EPOCH = 100
config.TRAIN.CONTINUE = False
config.TRAIN.FACTOR = 0.8
config.TRAIN.PATIENCE = 20
config.TRAIN.SHUFFLE = True

# test
config.TEST = edict()
config.TEST.BATCH_SIZE = 16
config.TEST.RECALL = []
config.TEST.TIOU = []
config.TEST.EVAL_TRAIN = False
config.TEST.NMS_THRESH = 0.4
config.TEST.INTERVAL = 0.25


def _update_dict(cfg, value):
    for k, v in value.items():
        if k in cfg:
            if k == 'PARAMS':
                cfg[k] = v
            elif isinstance(v, dict):
                _update_dict(cfg[k],v)
            else:
                cfg[k] = v
        else:
            raise ValueError("{} not exist in config.py".format(k))


def update_config(config_file):
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(config[k], v)
                else:
                    config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))

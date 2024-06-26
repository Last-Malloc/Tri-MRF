"""
FrameAvgPool: 以conv1d方式 将视觉信息 shape[b, vis_fea_len, num_sample_clips] -> [b, vis_fea_len, num_clips+1]
"""
from lib.models.frame_modules.frame_pool import FrameAvgPool

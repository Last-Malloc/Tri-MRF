from torch import nn


class FrameAvgPool(nn.Module):
    """
    以conv1d方式 将视觉信息 shape[b, vis_fea_len, num_sample_clips] -> [b, vis_fea_len, num_clips+1]
    """
    def __init__(self, cfg):
        super(FrameAvgPool, self).__init__()
        kernel_size = cfg.KERNEL_SIZE
        stride = cfg.STRIDE
        self.avg_pool = nn.AvgPool1d(kernel_size, stride, int(kernel_size/2))

    def forward(self, visual_input):
        vis_h = self.avg_pool(visual_input)
        return vis_h

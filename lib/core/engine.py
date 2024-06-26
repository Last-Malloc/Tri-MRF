import torch


class Engine(object):
    """
    模板文件，不推荐进行任何更改
    """
    def __init__(self):
        self.hooks = {}

    def hook(self, name, state):
        if name in self.hooks:
            self.hooks[name](state)

    def train(self, network, iterator, maxepoch, optimizer, scheduler):
        """
        可提供(不提供则不执行，不会报错):
            on_start: 推荐在该函数中执行model.train()等
            on_start_epoch:
            on_sample: 推荐在该函数中将数据从cpu->cuda
            on_forward: 在该函数中对模型返回loss、output进行操作，例如梯度裁剪等
            on_update:
            on_end_epoch: 推荐在该函数中执行scheduler更新optimizer的lr等
            on_end:

        流程：
            【初始化state：state = {'network': network, 'iterator': iterator, 'maxepoch': maxepoch, 'optimizer': optimizer,
                                'scheduler': scheduler, 'epoch': 0, 't': 0, 'train': True}
                        epoch为当前epoch数int范围[0, max_epoch]，等于时即为最终状态；
                        t为当前总iter数int范围[0, max_epoch * num_iters_per_epoch]，等于时即为最终状态
                            drop_last = False时，num_iters_per_epoch = ceil(len_dataset / batch_size)
                            drop_last = True时，num_iters_per_epoch = floor(len_dataset / batch_size)
                        可修改或添加内容，最终将返回state】
            【执行on_start】
            for 每个epoch:
                【执行on_start_epoch】
                for 每个iter(此iter数据为sample):
                    state['sample'] = sample
                    【执行on_sample】
                    执行模型
                    自动梯度求导
                    【执行on_forward】
                    优化器通过梯度下降更新参数
                    【执行on_update】
                【执行on_end_epoch】
            【执行on_end】

        :param network: 网络模型 【class/function】 需返回loss、output两部分，将临时赋值给state['output']、state['loss']，
                            在on_forward中对其操作后(可选)，state['output']、state['loss']将被赋值为None
                        若为class: 模型直接返回loss、output两部分
                        若为function: 在函数中调用模型，利用模型返回值构建成loss、output两部分
        :param iterator: dataloader 【class】
        :param maxepoch: max epoch【int】
        :param optimizer: 优化器 【class】
        :param scheduler: 调度器 【class】
        :return: state
        """
        state = {
            'network': network,
            'iterator': iterator,
            'maxepoch': maxepoch,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'epoch': 0,
            't': 0,
            'train': True,
        }

        self.hook('on_start', state)
        while state['epoch'] < state['maxepoch']:
            self.hook('on_start_epoch', state)
            for sample in state['iterator']:
                state['sample'] = sample
                self.hook('on_sample', state)

                def closure():
                    loss, output = state['network'](state['sample'])
                    state['output'] = output
                    state['loss'] = loss
                    loss.backward()
                    self.hook('on_forward', state)
                    state['output'] = None
                    state['loss'] = None
                    return loss
                state['optimizer'].zero_grad()
                state['optimizer'].step(closure)

                self.hook('on_update', state)
                state['t'] += 1
            state['epoch'] += 1
            self.hook('on_end_epoch', state)
        self.hook('on_end', state)
        return state

    def test(self, network, iterator, split):
        """
        可提供(不提供则不执行，不会报错):
            on_test_start: 推荐在该函数中执行model.eval()等
            on_test_sample: 推荐在该函数中将数据从cpu->cuda
            on_test_forward: 在该函数中对loss、output进行操作，例如将所有output存储起来等
            on_test_end

        流程：
            【初始化state：state = {'network': network, 'iterator': iterator, 'split':split, 't': 0, 'train': False}
                        t为当前测试总iter数int范围[0, num_iters_per_epoch]，等于时即为最终状态
                            drop_last = False时，num_iters_per_epoch = ceil(len_dataset / batch_size)
                            drop_last = True时，num_iters_per_epoch = floor(len_dataset / batch_size)
                        可修改或添加内容，最终将返回state】
            【执行on_test_start】
             for 每个iter(此iter数据为sample):
                 state['sample'] = sample
                 【执行on_test_sample】
                 执行模型
                 【执行on_test_forward】
            【执行on_test_end】
        :param network: 网络模型 【class/function】 需返回loss、output两部分，将临时赋值给state['output']、state['loss']，
                            在on_forward中对其操作后(可选)，state['output']、state['loss']将被赋值为None
                        若为class: 模型直接返回loss、output两部分
                        若为function: 在函数中调用模型，利用模型返回值构建成loss、output两部分
        :param iterator: dataloader【class】
        :param split: train/val/test【str】
        :return: state
        """
        state = {
            'network': network,
            'iterator': iterator,
            'split': split,
            't': 0,
            'train': False,
        }

        with torch.no_grad():
            self.hook('on_test_start', state)
            for sample in state['iterator']:
                state['sample'] = sample
                self.hook('on_test_sample', state)

                def closure():
                    loss, output = state['network'](state['sample'])
                    state['output'] = output
                    state['loss'] = loss
                    self.hook('on_test_forward', state)
                    state['output'] = None
                    state['loss'] = None
                closure()

                state['t'] += 1
            self.hook('on_test_end', state)
        return state

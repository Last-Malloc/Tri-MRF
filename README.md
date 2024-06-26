# Tri-MRF

This is the code for the paper "**Data, Content, Target-Oriented: A 3-Level Human-Like Framework for Moment Retrieval**". We appreciate the contribution of [2D-TAN](https://github.com/microsoft/2D-TAN) and [MSAT](https://github.com/mxingzhang90/MSAT/tree/2cfb646f84a32abf5a624fec73342a9d49641db8).

## Prerequisites

- python 3.7
- cuda 11.3
- pytorch 1.11.0
- torchvision 0.12.0
- torchtext 0.12.0
- pyyaml 6.0
- easydict 1.10
- h5py 3.7.0
- terminaltables 3.1.10

You can use the following commands in anaconda to create the same environment as ours:

```bash
conda create -n torch111 python=3.7
conda activate torch111
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install torchtext==0.12.0
pip install pyyaml==6.0
pip install easydict==1.10
pip install h5py==3.7.0
pip install terminaltables==3.1.10
```


## Quick Start

Please download the visual features from [box](https://rochester.box.com/s/8znalh6y5e82oml2lr7to8s6ntab6mav) or [dropbox](https://www.dropbox.com/sh/dszrtb85nua2jqe/AABGAEQhPtqBIRpGPY3gZey6a?dl=0) and save it to the `data/` folder. 

#### Training

**Please train on 2 gpus with the specific ''batch_size'' to keep the best performance, otherwise you shuld carefully change the learning rate.**

**Test device:  2 Nvidia RTX3090**
***OS version: Ubuntu 20.04***

Use the following commands for training:
```bash
# For TACoS
python moment_localization/train.py --cfg experiments/tacos/MSAT-128.yaml

# For ActivityNet Captions
python moment_localization/train.py --cfg experiments/activitynet/MSAT-32.yaml

# For Charades-STA
python moment_localization/train.py --cfg experiments/charades/MSAT-64.yaml
```

#### Testing
Our trained model are provided in [Baidu Netdisk](https://pan.baidu.com/s/19RDTaXsFpQbZ1iOtbufyHQ?pwd=mng7)(access code:mng7). Please download them to the `checkpoints` folder.

Then, run the following commands for evaluation: 
```bash
# For TACoS
python moment_localization/test.py --cfg experiments/tacos/MSAT-128.yaml --split test

# For ActivityNet Captions
python moment_localization/test.py --cfg experiments/activitynet/MSAT-32.yaml --split test

# For Charades-STA
python moment_localization/test.py --cfg experiments/charades/MSAT-64.yaml --split test
```

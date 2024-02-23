# neosr training

clone the neosr repository, branch is hpu:
```bash
git clone -b hpu https://github.com/krickwix/neosr.git
cd neosr/dataset
```

Retrieving the dataset:
```bash
pip install gdown
gdown 138oF5R56lumostbmARNQWUxV0R_v5heN
tar xf neosr_dataset.tar.gz
```

The dataset contains ground truth and low resolution images for training: nomos_uni_gt and nomos_uni_lr
It also contains ground truth and low res images for validation: val/gt and val/lr

Installing requirements:
```bash
cd ..
pip install .
```

Starting a training using LAZY_MODE (default):
```bash
python train.py -opt options/train_omnisr.yml
```
With defauts, python segfaults

Starting a training again
```bash
PT_HPU_LAZY_MODE=0 python train.py -opt options/train_omnisr.yml
```
The training should start. It is possible to follow the training progress using tensorboard:
```bash
tensorboard --logdir experiments/tb_logger
```
Every 1000 iterations, the model will be saved and validated and result of the inference will be visible from: ```experiments/train_omnisr/visualization```





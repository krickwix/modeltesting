# Simple GAN

## Dataset

Download the dataset tarball
```bash
pip install gdown
gdown 1YDW4GpwmUYeBooZdM2-y63r6pnjYQctt
```
Extract the tarball
```bash
tar xf img_align_celeba.tar
```

## Weights & Biases

https://wandb.ai/site

Put an API key at line:
```python
wandb.login(key='YOUR API KEY HERE')
```

## Start Training

```
python train_face_gan.py
```


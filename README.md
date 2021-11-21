# Unofficial PyTorch implementation of [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)

This repository is built upon [BEiT](https://github.com/microsoft/unilm/tree/master/beit), thanks very much!


Now, we implement the pretrain and finetune process according to the paper, but still **can't guarantee** the performance reported in the paper can be reproduced! 

## Difference

### `shuffle` and `unshuffle`

`shuffle` and `unshuffle` operations don't seem to be directly accessible in pytorch, so we use another method to realize this process:
+ For `shuffle`, we use the method of randomly generating mask-map (14x14) in BEiT, where `mask=0` illustrates keeping the token, `mask=1` denotes dropping the token (not participating caculation in encoder). Then all visible tokens (`mask=0`) are fed into encoder network.
+ For `unshuffle`, we get the postion embeddings (with adding the shared mask token) of all masked tokens according to the mask-map and then concate them with the visible tokens (from encoder), and feed them into the decoder network to recontrust.

### sine-cosine positional embeddings

The positional embeddings mentioned in the paper are `sine-cosine` version. And we adopt the implemention of [here](https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31), but it seems like a 1-D embeddings not 2-D's. So we don't know what effect it will bring.


## TODO
- [x] implement the finetune process
- [ ] reuse the model in `modeling_pretrain.py`
- [x] caculate the normalized pixels target
- [ ] add the `cls` token in the encoder
- [x] visualization of reconstruction image
- [ ] knn and linear prob
- [ ] ...

## Setup

```
pip install -r requirements.txt
```

## Run
1. Pretrain
```bash
# Set the path to save checkpoints
OUTPUT_DIR='output/pretrain_mae_base_patch16_224'
# path to imagenet-1k train set
DATA_PATH='/path/to/ImageNet_ILSVRC2012/train'


# batch_size can be adjusted according to the graphics card
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 run_mae_pretraining.py \
        --data_path ${DATA_PATH} \
        --mask_ratio 0.75 \
        --model pretrain_mae_base_patch16_224 \
        --batch_size 128 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 40 \
        --epochs 1600 \
        --output_dir ${OUTPUT_DIR}
```

2. Finetune
```bash
# Set the path to save checkpoints
OUTPUT_DIR='output/'
# path to imagenet-1k set
DATA_PATH='/path/to/ImageNet_ILSVRC2012'
# path to pretrain model
MODEL_PATH='/path/to/pretrain/checkpoint.pth'

# batch_size can be adjusted according to the graphics card
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 run_class_finetuning.py \
    --model vit_base_patch16_224 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 128 \
    --opt adamw \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 100 \
    --dist_eval
```
3. Visualization of reconstruction
```bash
# Set the path to save images
OUTPUT_DIR='output/'
# path to image for visualization
IMAGE_PATH='files/ILSVRC2012_val_00031649.JPEG'
# path to pretrain model
MODEL_PATH='/path/to/pretrain/checkpoint.pth'

# Now, it only supports pretrained models with normalized pixel targets
python run_mae_vis.py ${IMAGE_PATH} ${OUTPUT_DIR} ${MODEL_PATH}
```

## Result

|   model  | pretrain | finetune | accuracy | log | weight |
|:--------:|:--------:|:--------:|:--------:| :--------:|:--------:|
| vit-base |   400e   |   100e   |   83.1%  | [pretrain](files/pretrain_base_0.75_400e.txt) [finetune](files/pretrain_base_0.75_400e_finetune_100e.txt)| [Google drive](https://drive.google.com/drive/folders/182F5SLwJnGVngkzguTelja4PztYLTXfa?usp=sharing) |

Due to the limited gpus, it's really a chanllenge for us to pretrain with larger model or longer schedule mentioned in the paper. 

So if one can fininsh it, please feel free to report it in the issue or push a PR, thank you!

And your star is my motivation, thank u~

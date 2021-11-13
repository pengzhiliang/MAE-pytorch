
# Unofficial PyTorch implementation of [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)

This repository is built upon [BEiT](https://github.com/microsoft/unilm/tree/master/beit), thanks very mush!


Now, we only implement the pretrain process according to the paper.

## TODO
1. implement the finetune process
2. reuse the model in `modeling_pretrain.py`
3. caculate the normalized pixels target
4. add the `cls` token in the encoder
5. ...

## Setup

```
pip install -r requirements.txt
```

## Run

```bash
# Set the path to save checkpoints
OUTPUT_DIR='output/'
# Download and extract ImageNet-22k
DATA_PATH='../ImageNet_ILSVRC2012/train'


OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 run_mae_pretraining.py \
        --data_path ${DATA_PATH} \
        --mask_ratio 0.75 \
        --model pretrain_mae_base_patch16_224 \
        --batch_size 128 \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 40 \
        --epochs 1600 \
        --output_dir ${OUTPUT_DIR}
```

Note: the pretrain result is on the way ~
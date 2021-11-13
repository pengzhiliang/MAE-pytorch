
# Unofficial PyTorch implementation of [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)

This repository is built upon [BEiT](https://github.com/microsoft/unilm/tree/master/beit), thanks very much!


Now, we only implement the pretrain process according to the paper, and **can't guarantee** the performance reported in the paper can be reproduced!

## Difference
At the same time, `shuffle` and `unshuffle` operations don't seem to be directly accessible in pytorch, so we use another method to realize this process:
+ For `shuffle`, we used the method of randomly generating mask-map (14x14) in BEiT, where `mask=0` illustrates keep the token, `mask=1` denotes drop the token (not participating caculation in Encoder). Then all visible tokens (`mask=0`) are put into encoder network.
+ For `unshuffle`, we get the postion embeddings (with adding the shared mask token) of all mask tokens according to the mask-map and then concate them with the visible tokens (from encoder), and put them into the decoder network to recontrust.

## TODO
- [ ] implement the finetune process
- [ ] reuse the model in `modeling_pretrain.py`
- [x] caculate the normalized pixels target
- [ ] add the `cls` token in the encoder
- [ ] ...

## Setup

```
pip install -r requirements.txt
```

## Run

```bash
# Set the path to save checkpoints
OUTPUT_DIR='output/'
# path to imagenet-1k train set
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

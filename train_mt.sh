#!/bin/bash

# wandb credentials
!wandb login b8d75bf0df76638730fd957d848903e52b4713bc

# tokenizer model
model=$1

rootdir='./data'
rawdir=$rootdir/'raw'
clndir=$rootdir/'cleaned'
datadir=$rootdir/'models'/$model
tokdir=$datadir/'tokenized'
predir=$datadir/'preprocessed'
chkdir=$datadir/'checkpoints'
evldir=$datadir/'eval'

# fairseq-preprocess
fairseq-preprocess --source-lang hi --target-lang te \
    --trainpref $tokdir/train --validpref $tokdir/valid --testpref $tokdir/test \
    --workers 8 \
    --destdir $predir

# fairseq-train
fairseq-train $predir \
    --source-lang='hi' --target-lang='te' \
    --save-dir $chkdir \
    # --wandb-project 'trieceps_train_mt' \
    --max-source-positions=210 \
    --max-target-positions=210 \
    --save-interval=1 \
    --arch=transformer \
    --criterion=label_smoothed_cross_entropy \
    --lr-scheduler=inverse_sqrt \
    --label-smoothing=0.1 \
    --optimizer adam \
    --adam-betas "(0.9, 0.98)" \
    --clip-norm 1.0 \
    --warmup-init-lr 1e-07 \
    --lr 0.0005 \
    --warmup-updates 4000 \
    --dropout 0.2 \
    --max-epoch 5 \
    --no-last-checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --distributed-world-size 4 \
    --max-tokens 256

# fairseq-generate
fairseq-generate $predir \
    --path $chkdir/checkpoint_5.pt \
    --batch-size 128 --beam 5 > $evldir/'test_inference.txt'
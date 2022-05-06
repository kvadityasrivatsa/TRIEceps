#!/bin/bash

# wandb credentials
# wandb login b8d75bf0df76638730fd957d848903e52b4713bc

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

max_epochs=5

# fairseq-preprocess
fairseq-preprocess --source-lang hi --target-lang te \
    --trainpref $tokdir/train --validpref $tokdir/valid --testpref $tokdir/test \
    --workers 8 \
    --destdir $predir

#    --wandb-project 'trieceps_train_mt' \

# fairseq-train
fairseq-train $predir \
    --source-lang='hi' --target-lang='te' \
    --save-dir $chkdir \
    --encoder-embed-dim 300 --decoder-embed-dim 300 \
    --encoder-ffn-embed-dim 1000 --decoder-ffn-embed-dim 1000 \
    --encoder-layers 3 --decoder-layers 3 \
    --encoder-attention-heads 5 --decoder-attention-heads 5 \
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
    --max-epoch $max_epochs \
    --no-last-checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --distributed-world-size 4 \
    --max-tokens 256 \
    --seed 0

# fairseq-generate
echo "Running inference"
fairseq-generate $predir \
    --path $chkdir/checkpoint$max_epochs.pt \
    --batch-size 64 --beam 5 > $evldir/'test_inference.txt'

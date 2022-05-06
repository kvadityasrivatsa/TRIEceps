#!/bin/bash

# tokenizer model
model=$1

# model checkpoint {1,2,3,4,5,best}
chp=$2

rootdir='./data'
rawdir=$rootdir/'raw'
clndir=$rootdir/'cleaned'
datadir=$rootdir/'models'/$model
tokdir=$datadir/'tokenized'
predir=$datadir/'preprocessed'
chkdir=$datadir/'checkpoints'
evldir=$datadir/'eval'

# fairseq-generate
fairseq-generate $predir \
    --path $chkdir/checkpoint_$chp.pt \
    --batch-size 128 --beam 5 > $evldir/'test_inference_$chp.txt'

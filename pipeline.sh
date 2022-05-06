#!/bin/bash

model=$1

rootdir='./data'
rawdir=$rootdir/'raw'
clndir=$rootdir/'cleaned'; #rm -rf $clndir; mkdir -p $clndir
datadir=$rootdir/'models'/$model
tokdir=$datadir/'tokenized'; rm -rf $tokdir; mkdir -p $tokdir
predir=$datadir/'preprocessed'; rm -rf $predir
chkdir=$datadir/'checkpoints'; rm -rf $chkdir
evldir=$datadir/'eval'; rm -rf $evldir; mkdir -p $evldir

# cleaner.py
#python3 cleaner.py 5000000

# tokenize.py
python3 tokenizer.py $model

# train_mt.py
bash train_mt.sh $model

# evaluate.py


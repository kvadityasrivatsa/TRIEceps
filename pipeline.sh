#!/bin/bash

model=$1

rootdir='./data'
rawdir=$rootdir/'raw'
clndir=$rootdir/'cleaned'; rm -rf $clndir
tokdir=$datadir/'tokenized'; rm -rf $tokdir
predir=$datadir/'preprocessed'; rm -rf $predir
chkdir=$datadir/'checkpoints'; rm -rf $chkdir
evldir=$datadir/'eval'; rm -rf $evldir

# cleaner.py

# tokenize.py

# train_mt.py
bash train_mt.sh $model

# evaluate.py
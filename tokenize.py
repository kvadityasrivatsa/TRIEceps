import os
import re
import sys
from utils import *
#from tqdm import tqdm
import sentencepiece as spm

model = str(sys.argv[1]).strip()

rootdir='./data'
rawdir=os.path.join(rootdir,'raw')
clndir=os.path.join(rootdir,'cleaned')
datadir=os.path.join(rootdir,'models',model)
tokdir=os.path.join(datadir,'tokenized')

def run_sentencepiece(lang,model_type,model_prefix):

    spm.SentencePieceTrainer.train(f'--input={clndir}/train.{lang} \
                                    --model_prefix={tokdir}/{model_prefix}.{lang} \
                                    --vocab_size=2000 --model_type={model_type}')

    sp = spm.SentencePieceProcessor()
    sp.load(f'{tokdir}/{model_prefix}.{lang}.model')

    for f in sorted(os.listdir(clndir)):
        if not f.endswith(f'.{lang}'):
            continue
        fpath = os.path.join(clndir,f)
        text = read_txt(fpath)
        print(f'tokenizing {fpath}')
        text = [' '.join(sp.encode_as_pieces(l)) for l in text]
        tpath = os.path.join(tokdir,f)
        write_txt(text,tpath)
        print(f'tokenized. saving to {tpath}\n')

run_sentencepiece('hi','bpe',model)
run_sentencepiece('te','bpe',model)

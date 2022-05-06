import os
import re
import sys
from utils import *
#from tqdm import tqdm
# import sentencepiece as spm
from tokenizers import ByteLevelBPETokenizer, CharBPETokenizer, SentencePieceBPETokenizer, BertWordPieceTokenizer

model = str(sys.argv[1]).strip()

rootdir='./data'
rawdir=os.path.join(rootdir,'raw')
clndir=os.path.join(rootdir,'cleaned')
datadir=os.path.join(rootdir,'models',model)
tokdir=os.path.join(datadir,'tokenized')

# def run_sentencepiece(lang,model_type,model_prefix):

#     spm.SentencePieceTrainer.train(f'--input={clndir}/train.{lang} \
#                                     --model_prefix={tokdir}/{model_prefix}.{lang} \
#                                     --vocab_size=30000 --model_type={model_type}')

#     sp = spm.SentencePieceProcessor()
#     sp.load(f'{tokdir}/{model_prefix}.{lang}.model')

#     for f in sorted(os.listdir(clndir)):
#         if not f.endswith(f'.{lang}'):
#             continue
#         fpath = os.path.join(clndir,f)
#         text = read_txt(fpath)
#         print(f'tokenizing {fpath}')
#         text = [' '.join(sp.encode_as_pieces(l)) for l in text]
#         tpath = os.path.join(tokdir,f)
#         write_txt(text,tpath)
#         print(f'tokenized. saving to {tpath}\n')

def run_baseline(lang,model):

    if model == 'byte_bpe':
        tokenizer = ByteLevelBPETokenizer()
    elif model == 'char_bpe':
        tokenizer = CharBPETokenizer()
    elif model == 'sentencepiece_bpe':
        tokenizer = SentencePieceBPETokenizer()
    elif model == 'wordpiece_bpe':
        tokenizer = BertWordPieceTokenizer()

    tokenizer.train([f'{clndir}/train.{lang}'],vocab_size=30000)

    for f in sorted(os.listdir(clndir)):
        if not f.endswith(f'.{lang}'):
            continue
        fpath = os.path.join(clndir,f)
        text = read_txt(fpath)
        print(f'tokenizing {fpath}')
        enc_text = tokenizer.encode_batch(text,add_special_tokens=False)
        tpath = os.path.join(tokdir,f)
        write_txt(enc_text,tpath)
        print(f'tokenized. saving to {tpath}\n')

    tokenizer.save(f'{tokdir}/{model}.{lang}.tok')
    

run_baseline('hi',model)
run_baseline('te',model)

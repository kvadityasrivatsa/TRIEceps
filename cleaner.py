import os
import re
import sys
from utils import *

rootdir='./data'
rawdir=os.path.join(rootdir,'raw')
clndir=os.path.join(rootdir,'cleaned')

def clean(line):
    return line.strip()

for f in os.listdir(rawdir):
    fpath = os.path.join(rawdir,f)
    print(f'cleaning {fpath}')
    text = read_txt(fpath)
    tpath = os.path.join(clndir,f)
    print(f'cleaned. saving to {tpath}\n')
    write_txt(text,tpath)
    
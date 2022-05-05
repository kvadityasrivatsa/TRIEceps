import os
import re
import sys
from sklearn.model_selection import train_test_split as tts

from utils import *

rootdir='./data'
rawdir=os.path.join(rootdir,'raw')
clndir=os.path.join(rootdir,'cleaned')

nsample = int(sys.argv[1])

def clean(line):
    return line.strip()

text_hi = [clean(l) for l in read_txt(os.path.join(rawdir,'orig.hi'))[:nsample]]
text_te = [clean(l) for l in read_txt(os.path.join(rawdir,'orig.te'))[:nsample]]

text = zip(text_hi,text_te)
train_text, aux_text = tts(text,test_size=0.2)
test_text, valid_text = tts(aux_text,test_size=0.5)

print(len(text))
print(len(train_text),len(valid_text),len(test_text))

train_hi, train_te = zip(*train_text)
valid_hi, valid_te = zip(*valid_text)
test_hi, test_te = zip(*test_text)

write_txt(train_hi,os.path.join(clndir,'train.hi'))
write_txt(train_te,os.path.join(clndir,'train.te'))
write_txt(valid_hi,os.path.join(clndir,'valid.hi'))
write_txt(valid_te,os.path.join(clndir,'valid.te'))
write_txt(test_hi,os.path.join(clndir,'test.hi'))
write_txt(test_te,os.path.join(clndir,'test.te'))



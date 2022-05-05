import os
import shutil
#from tqdm import tqdm

def read_txt(path):
    print(f'reading from {path}')
    data = []
    with open(path,'r') as f:
        for l in f:
            data.append(l.strip())
    print('read completed.\n')
    return data

def write_txt(data,path):
    print(f'writing to {path}')
    with open(path,'w') as f:
        f.writelines([l.strip()+'\n' for l in data])
    print('write completed.')

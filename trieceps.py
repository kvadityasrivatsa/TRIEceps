import re
import pickle
import numpy as np
import pandas as pd
from plotly import express as px
from tqdm import tqdm
from collections import Counter

class Node():
    def __init__(self,char,prev):
        self.char = char
        self.prev = prev
        self.next = {}
        self.pass_count = 0
        self.end_count = 0
#         self.tokens = Counter()
        
class Trie():
    def __init__(self,reverse=False):
        self.tree = {}
        self.reverse = reverse 
        
    def add(self,word):
        w = word[::-1] if self.reverse else word
        
        # for first char
        c = w[0]
        if c not in self.tree:
            self.tree[c] = Node(c,None)
        cnode = self.tree[c]
        cnode.pass_count += 1
#         cnode.tokens.update([w])
        
        # for the rest
        if len(w) > 1:
            for l, c in enumerate(w[1:]):
                if c not in cnode.next:
                    cnode.next[c] = Node(c,cnode)
                cnode = cnode.next[c]
                cnode.pass_count += 1
#                 cnode.tokens.update([w])
                
        cnode.end_count += 1
            
    def display(self,depth,edge_thresh=1,max_edges=5):
        
        file = ['digraph G {','size ="100,100";']
        
        def walk(d,nc,node,edge_thresh=1):
            if d==depth:
                return nc
            ncl = nc
            
            node_next_valid = [c for c in node.next if node.next[c].pass_count >= edge_thresh]
            node_next_valid = [v[1] for v in sorted([[node.next[c].pass_count, c] for c in node_next_valid],reverse=True)][:max_edges]
            for i, c in enumerate(node_next_valid):
                if node.next[c].pass_count < edge_thresh:
                    continue
                ncl += 1
                file.append(f'\t{ncl} [label="{c}"];')
                file.append(f'\t{nc} -> {ncl} [label="{node.next[c].pass_count}"];')
                ncl = walk(d+1,ncl,node.next[c],edge_thresh=edge_thresh)
            return ncl
        
        nc = 0
        file.append(f'\t{nc} [label="_"];')
        ncl = nc
        for c, _ in list(sorted([[c,self.tree[c].pass_count] for c in self.tree if self.tree[c].pass_count >= edge_thresh],reverse=True)):#[:max_edges]:
            ncl += 1
            file.append(f'\t{ncl} [label="{c}"];')
            file.append(f'\t{nc} -> {ncl} [label="{self.tree[c].pass_count}"];')
            ncl = walk(0,ncl,self.tree[c],edge_thresh=edge_thresh)
        
        file.append('}')
        file = [l+'\n' for l in file]
        open(f'trieceps_d{depth}.gv','w').writelines(file)
            
            
    def find(self,st,edge_thresh=1):
        
        def walk(i,st,node):
            if i==len(st):
                return node, [node.pass_count, node.end_count, len([c for c in node.next if node.next[c].pass_count >= edge_thresh])]
            c = st[i]
            if c in node.next:
                return walk(i+1,st,node.next[c])
            else:
                return None, [0,0,0]
        return walk(1,st,self.tree[st[0]])
    
    def find_return_char_level_stats(self,st):
        
        def walk(i,st,node):
            if i==len(st):
                return [node]
            return [node] + walk(i+1,st,node.next[st[i]])
        
        return walk(1,st,self.tree[st[0]])

class TRIEceps():

    def __init__(self,lang,split_thresh=0.09,max_splits=2):
        self.lang = lang
        self.f_trie, self.r_trie = Trie(reverse=False), Trie(reverse=True)
        if lang=='hi':
            self.lang_start, self.lang_end, self.lang_reg = 'ऀ', 'ॿ', r'[^0-9\u0900-\u0957]'
        elif lang=='te':
            self.lang_start, self.lang_end, self.lang_reg = 'ఀ', '౿', r'[^0-9\u0C00-\u0C7F]'
        elif lang=='en':
            self.lang_start, self.lang_end, self.lang_reg = 'a', 'z', r'[^a-zA-Z0-9]'

        self.split_thresh = split_thresh  # higher -> less splitting tendency
        self.max_splits = max_splits

    def build_tries(self,path,line_count=None,max_token_length=20):
        
        line_count = line_count if line_count!=None else len(open(path,'r').read().split('\n'))-1
        line_idx = 0
        pbar = tqdm(total=line_count)
        with open(path,'r') as f:
            for l in f:
                if line_idx == line_count:
                    break
                l = re.sub(r"[ ]+",' ',re.sub(self.lang_reg,' ',l.strip())).split()
                for w in l:
                    if len(w) > max_token_length:
                        continue
                    w = str(w)
                    self.f_trie.add(w)
                    self.r_trie.add(w)
                line_idx += 1
                pbar.update(1)

    def candidacy_split(self,w):

        def disect(w,brk):
            dis = []
            brk.append(len(w)-1)
            p = 0
            for b in brk:
                dis.append(w[p:b+1])
                p = b+1
            return dis
        
        def norm_std(z):
            z = z/np.sum(z)
            return np.std(z)
        try:
            fwd_nodes = self.f_trie.find_return_char_level_stats(w)
            rev_nodes = self.r_trie.find_return_char_level_stats(w[::-1])[::-1]
        except:
            return list(w)
        voc_freq = fwd_nodes[-1].end_count
        _split_thresh = self.split_thresh * (np.log(voc_freq + 1) + 1)
        
        fwd = [nc.pass_count*len(nc.next)/(np.log(norm_std([nc.next[ncl].pass_count for ncl in nc.next])+1)+1) for nc in fwd_nodes]
        rev = [nc.pass_count*len(nc.next)/(np.log(norm_std([nc.next[ncl].pass_count for ncl in nc.next])+1)+1) for nc in rev_nodes]
        com = [fwd[i]*rev[i+1] for i in range(len(w)-1)]
        com = com/np.sum(com)
        brk = sorted([i for i,s in sorted(enumerate(com),
                                key=lambda x: x[1],
                                reverse=True) if s>=_split_thresh][:self.max_splits])
        if len(brk)==0:
            return [w]
        return disect(w,brk)

    def candidate_split_batch(self,text):
        cache = {}
        tokenized_text = []
        for s in tqdm(text):
            tokenized_s = []
            for w in s.split():
                if w in cache:
                    tokenized_s.extend(cache[w])
                else:
                    subw = self.candidacy_split(w)
                    subw[0] = '▁'+subw[0]
                    cache[w] = subw.copy()
                    tokenized_s.extend(subw)
                    
            tokenized_text.append(' '.join(tokenized_s))
        return tokenized_text

    def train(self,path,line_count=None,max_token_length=20):
        self.build_tries(path,line_count=line_count,max_token_length=max_token_length)

    def encode(self,s):
        tokenized_s = []
        for w in s.split():
            subw = self.candidacy_split(w)
            subw[0] = '▁'+subw[0]
            tokenized_s.extend(subw)
        return tokenized_s

    def encode_batch(self,text):
        return self.candidate_split_batch(text)

    def decode(self,s):
        return re.sub('▁',' ',s).strip()

    def decode_batch(self,text):
        return [self.decode(s) for s in tqdm(text)]

    def save(self,path):
        pickle.dump({'f_trie':self.f_trie,
                    'r_trie':self.r_trie},
                    open(path,'wb'))

    def load(self,path):
        s_obj = pickle.load(open(path,'rb'))
        self.f_trie(s_obj['f_trie'])
        self.r_trie(s_obj['r_trie'])
# TRIEceps: TRIE Based Affixation-Informed Tokenization
The aim of this project is to utilize the TRIE data structure to automatically record and identify affixation (primarily suffixation) patterns from a given piece of text. We make use of these patterns to attempt to improve tokenization. Further, we apply the novel tokenizer on Hindi and Telugu data to train a Machine Translation (MT) pipeline, wherein we compare the performance against well-known tokenization methods.
## Getting Started
### Dependencies
Python >= 3.8
### Installation
```
pip install --editable .
```
### How to use
Fetching the data
```
$ cd TRIEceps/
$ gdown https://drive.google.com/uc?id=1WHE0IgHm_oFNW3X1C_Ax0AeLozGPLREt
$ unzip data.zip
```
Training MT pipeline on SentencePiece Baseline
```
$ bash train_mt.sh sentencepiece_bpe
```
Training MT pipeline on TRIEceps
```
$ bash train_mt.sh trieceps_candidacy
```
The processed data, checkpoints, and test outputs will be saved at `./data/models/{model_type}/`
## Pretrained Models
The pretrained checkpoints and processed data can be found [here](https://drive.google.com/file/d/1WHE0IgHm_oFNW3X1C_Ax0AeLozGPLREt/view?usp=sharing). 
Evaluation files can be found [here](https://drive.google.com/file/d/1rD-4oM2XQ13xP-RyPvhJEhyWdmBKntGy/view?usp=sharing).

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
$ gdown LINK
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
The pretrained checkpoints as well as the evaluation test output files can be found here.

## License

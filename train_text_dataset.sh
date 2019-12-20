#!/bin/bash

DATASET_PATH=../../dataset/mard
LANG=en
DIC_PATH=/usr/local/lib/mecab/dic/mecab-ipadic-neologd
# OUTPUT_PATH=model/corpus-w2v-model/word2vec.gensim.model
OUTPUT_PATH=model/mard-glove-model/
PRETRAINED_MODEL_PATH=model/enwiki-w2v-model/word2vec.gensim.model
SIZE=50
WINDOW=15
MIN_COUNT=5
EPOCH=25

python train_text_dataset.py -o $OUTPUT_PATH --dictionary-path=$DIC_PATH --dataset-path=$DATASET_PATH --lang=$LANG --size=$SIZE --window=$WINDOW --min-count=$MIN_COUNT --epoch=$EPOCH

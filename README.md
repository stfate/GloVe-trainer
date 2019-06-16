GloVe-trainer
=============

# 概要
Wikipediaもしくは任意のテキストコーパスから[GloVe](https://nlp.stanford.edu/projects/glove/)の学習を行うPythonスクリプト．

# 使い方
1. bin/以下にGloVeのバイナリ一式(cooccur/glove/shuffle/vocab_count)を配置する．
2. train_text_dataset.sh / train_wikipedia.shを実行する．

## Wikipediaの学習

```bash
python src/train_wikipedia.py -o $OUTPUT_PATH --dictionary-path=$DIC_PATH --wikipedia-dump-path=$WIKIPEDIA_DUMP_PATH --lang=$LANG --size=$SIZE --window=$WINDOW --min-count=$MIN_COUNT --epoch=$EPOCH
```

## 任意のテキストデータの学習

```bash
python src/train_text_dataset.py -o $OUTPUT_PATH --dictionary-path=$DIC_PATH --dataset-path=$DATASET_PATH --lang=$LANG --size=$SIZE --window=$WINDOW --min-count=$MIN_COUNT --epoch=$EPOCH
```

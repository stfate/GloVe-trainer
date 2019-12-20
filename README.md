GloVe-trainer
=============

# Overview

Wikipediaもしくは任意のテキストコーパスから[GloVe](https://nlp.stanford.edu/projects/glove/)の学習を行うPythonスクリプト．


# Requirements

- [GloVe](https://nlp.stanford.edu/projects/glove/)

他の依存パッケージは`requirements.txt`を参照．



# Setup

```bash
git submodule init
git submodule update
pip install -r requirements.txt
```


# Run

1. `bin/`以下にGloVeのバイナリ一式(cooccur/glove/shuffle/vocab_count)を配置する．
2. train_text_dataset.sh / train_wikipedia.shを実行する．

## 任意のテキストデータの学習

`TextDatasetBase`を継承したデータセットクラスを作成することで，任意のテキストデータセットに対し実行することが可能．

```python
class TextDatasetBase(ABC):
    """
    a bass class for text dataset
    
    Attributes
    ----------
    """
    @abstractmethod
    def iter_docs(self):
        """
        iterator of documents
        
        Parameters
        ----------
        """
        yield None
```

```bash
python train_text_dataset.py -o $OUTPUT_PATH --dictionary-path=$DIC_PATH --dataset-path=$DATASET_PATH --lang=$LANG --size=$SIZE --window=$WINDOW --min-count=$MIN_COUNT --epoch=$EPOCH
```

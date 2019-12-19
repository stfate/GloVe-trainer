import argparse
from functools import partial
import tempfile

import document_tokenizer
import text_dataset
import glove_trainer


def get_options():
    parser = argparse.ArgumentParser()

    parser.add_argument("-o", "--output-model-path", default="model/word2vec.gensim.model")
    parser.add_argument("--size", type=int, default=50)
    parser.add_argument("--window", type=int, default=15)
    parser.add_argument("--min-count", type=int, default=10)
    parser.add_argument("--epoch", type=int, default=25)

    parser.add_argument("--wikipedia-dump-path", default="data/jawiki-latest-pages-articles.xml.bz2")
    parser.add_argument("--lang", default="ja")

    parser.add_argument("--dictionary-path", default="output/dic")

    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    options = get_options()

    dic_path = options["dictionary_path"]
    wikipedia_dump_path = options["wikipedia_dump_path"]
    wikipedia = text_dataset.WikipediaDataset()

    output_model_path = options["output_model_path"]
    size = options["size"]
    window = options["window"]
    min_count = options["min_count"]
    epoch = options["epoch"]
    lang = options["lang"]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        iter_docs = partial(wikipedia.iter_docs, wikipedia_dump_path, temp_dir)
        if lang == "ja":
            tokenizer = document_tokenizer.MecabDocumentTokenizer(dic_path)
        elif lang == "en":
            tokenizer = document_tokenizer.NltkDocumentTokenizer()
        glove_trainer.train_glove_model(output_model_path, iter_docs, tokenizer, size, window, min_count, epoch)

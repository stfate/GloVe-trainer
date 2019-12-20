import functools
from pathlib import Path
import multiprocessing
import logging
import os
import tempfile

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s:%(message)s", datefmt="%Y/%m/%d %H:%M:%S")


def generate_input_text(iter_tokens, output_fname):
    with open(output_fname, "w") as fo:
        for tokens in iter_tokens():
            input_text = " ".join(tokens) + " "
            fo.write(input_text)


def train_glove_model(output_model_path, iter_docs, tokenizer, size=50, window=15, min_count=5, epoch=25):
    """
    Parameters
    ----------
    output_model_path : string
        Path of GloVe model
    iter_docs : iterator
        Iterator of documents, which are lists of words
    tokenizer : class
        Word tokenizer, which must have tokenize() method
    size : int
        vector size
    window : int
        window size
    min_count : int
        minimum count
    epoch : int
        number of epochs
    """
    # setup paths to GloVe binaries 
    cur_path = Path(__file__)
    bin_path = cur_path.parent / "bin"
    vocab_count_bin = bin_path / "vocab_count"
    cooccur_bin = bin_path / "cooccur"
    shuffle_bin = bin_path / "shuffle"
    glove_bin = bin_path / "glove"

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        input_fn = tmp_dir_path / "input.txt"
        vocab_fn = tmp_dir_path / "vocab.txt"
        cooccur_fn = tmp_dir_path / "cooccurence.txt"
        cooccur_shuffle_fn = tmp_dir_path / "cooccurrence_shuffle.txt"

        # generate input text
        logging.info("generate input text file...")
        iter_tokens = tokenizer.get_tokens_iterator(iter_docs, normalize=True)
        generate_input_text(iter_tokens, input_fn)

        # vocab_count
        logging.info("vocab_count...")
        cmd = f"{str(vocab_count_bin)} -min-count {min_count} -verbose 2 < {str(input_fn)} > {str(vocab_fn)}"
        os.system(cmd)

        # cooccur
        logging.info("cooccur...")
        cmd = f"{str(cooccur_bin)} -memory 4 -vocab-file {str(vocab_fn)} -verbose 2 -window-size {window} < {str(input_fn)} > {str(cooccur_fn)}"
        os.system(cmd)

        # shuffle
        logging.info("shuffle...")
        cmd = f"{str(shuffle_bin)} -memory 4 -verbose 2 < {str(cooccur_fn)} > {str(cooccur_shuffle_fn)}"
        os.system(cmd)

        # glove
        logging.info("glove...")
        output_dir = Path(output_model_path)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        output_fn = output_dir / "vectors"
        cmd = f"{str(glove_bin)} -save-file {str(output_fn)} -threads 2 -input-file {str(cooccur_shuffle_fn)} -x-max 10 -iter {epoch} -vector-size {size} -binary 2 -vocab-file {str(vocab_fn)} -verbose 2"
        os.system(cmd)

    logging.info("done.")

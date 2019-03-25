# Copyright (C) 2018-2019 Gianni Rosa Gallina. See LICENSE file.

from fastai.text import *
import argparse

def preprocess_text(dir_path, chunksize=24000, lang='en', max_vocab=60000, min_freq=2):
    print(f' dir_path: {dir_path}')
    print(f'chunksize: {chunksize}')
    print(f'max_vocab: {max_vocab}')
    print(f' min_freq: {min_freq}')
    print(f'     lang: {lang}')

    dir_path = Path(dir_path)
    assert dir_path.exists(), f'Error: {dir_path} does not exist.'
    
    df_trn = pd.read_csv(dir_path/'train.csv', header=None)
    df_val = pd.read_csv(dir_path/'val.csv', header=None)

    tokenizer = Tokenizer(lang=lang, n_cpus=8) # change CPUs according to your configuration
    
    data = TextLMDataBunch.from_df(dir_path, df_trn, df_val, tokenizer=tokenizer,
                                   text_cols=0, chunksize=chunksize, max_vocab=max_vocab,
                                   min_freq=min_freq)
    data.save()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_path', required=True,
                        help='the directory where the merged Wikipedia documents are stored')
    parser.add_argument('--chunksize', type=int, default=24000,
                        help='the number of documents to process in batch')
    parser.add_argument('--max_vocab', type=int, default=60000,
                        help='the number of unique tokens to support')
    parser.add_argument('--min_freq', type=int, default=2,
                        help='the minimum token frequency to consider')
    parser.add_argument('--lang', default="en",
                        help='the language of text documents to process')
    args = parser.parse_args()
    preprocess_text(args.dir_path, args.chunksize, args.lang, args.max_vocab, args.min_freq)
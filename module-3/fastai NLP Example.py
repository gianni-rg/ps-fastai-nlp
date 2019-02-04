#!/usr/bin/env python
# coding: utf-8

from fastai.text import *   # Quick access to NLP functionality

if __name__ == '__main__':
    path = untar_data(URLs.IMDB_SAMPLE)

    df = pd.read_csv(path/'texts.csv', header=None)
    df.head()

    data_lm = TextLMDataBunch.from_csv(path, 'texts.csv')
    data_clas = TextClasDataBunch.from_csv(path, 'texts.csv', vocab=data_lm.train_ds.vocab, bs=42)

    moms = (0.8,0.7)

    learn = language_model_learner(data_lm, pretrained_model=URLs.WT103_1)
    learn.unfreeze()
    learn.fit_one_cycle(4, slice(1e-2), moms=moms)

    learn.save_encoder('enc')

    learn = text_classifier_learner(data_clas)
    learn.load_encoder('enc')
    learn.freeze()
    learn.fit_one_cycle(4, moms=moms)

    learn.unfreeze()
    learn.fit_one_cycle(8, slice(1e-5,1e-3), moms=moms)
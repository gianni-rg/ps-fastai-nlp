#!/usr/bin/env python
# coding: utf-8

from fastai.text import *

DATA_PATH = Path('../datasets/20news')
DATA_PATH.mkdir(exist_ok=True)

bs = 32

# Comment these two lines if you have already done and saved the pre-processing
data_lm = (TextList.from_folder(DATA_PATH)
            .filter_by_folder(include=['20news-bydate-train', '20news-bydate-test']) 
            .random_split_by_pct(0.1) 
            .label_for_lm()
            .databunch(bs=bs))
data_lm.save('tmp_lm')

# Uncomment this line if you already done the previous pre-processing
#data_lm = TextLMDataBunch.load(DATA_PATH, 'tmp_lm', bs=bs)

learn = language_model_learner(data_lm, pretrained_model=URLs.WT103_1, drop_mult=0.3)

learn.fit_one_cycle(1, 1e-2, moms=(0.8,0.7))
learn.save('fit_head')

learn.unfreeze()
learn.fit_one_cycle(15, 1e-3, moms=(0.8,0.7))
learn.save('fine_tuned')

learn.save_encoder('fine_tuned_enc')

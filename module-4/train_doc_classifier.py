#!/usr/bin/env python
# coding: utf-8

from fastai.text import *

DATA_PATH = Path('../datasets/20news')
DATA_PATH.mkdir(exist_ok=True)

bs = 12

data_lm = TextLMDataBunch.load(DATA_PATH, bs=bs, cache_name='tmp_lm')

classes = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
           'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles',
           'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space',
           'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc',
           'talk.religion.misc']

# Comment these two lines if you have already done and saved the pre-processing
data_clas = TextClasDataBunch.from_folder(DATA_PATH, train='20news-bydate-train', valid='20news-bydate-test', classes=classes, vocab=data_lm.vocab)
data_clas.save('tmp_clas')

# Uncomment this line if you already done the previous pre-processing
#data_clas = TextClasDataBunch.load(DATA_PATH, 'tmp_clas', bs=bs)

drop_mult = 0.5

learn = text_classifier_learner(data_clas, drop_mult=drop_mult)
learn.load_encoder('fine_tuned_enc')
learn.freeze()

learn.fit_one_cycle(1, 5e-2, moms=(0.8,0.7))
learn.save('first')

learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))
learn.save('second')

learn.freeze_to(-3)
learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7))
learn.save('third')

learn.unfreeze()
learn.fit_one_cycle(2, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))
learn.save('final')

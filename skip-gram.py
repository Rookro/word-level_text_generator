#!/usr/bin/env python
'''
word2vec(skip-gram) in gensim
'''

import logging
import os
from gensim.models import word2vec

logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)

cur_dir = os.path.join(os.path.dirname(__file__))
sentences = word2vec.Text8Corpus(os.path.join(cur_dir, "corpus.juman"))
model = word2vec.Word2Vec(sentences, size = 300, min_count = 3, window = 15, iter = 8, sg = 1, workers = 8)

#model.int_sims(replace = True)
model.save("skipgram_gensim")

model = word2vec.Word2Vec.load("skipgram_gensim")




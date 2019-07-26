import pandas as pd
import numpy as np
import gc
from gensim.models import Word2Vec
from datetime import datetime
import random
from tqdm import tqdm

class Item2vec:

    def __init__(self,sentences,epochs=10,size=16,sg=1,n_jobs=-1,shuffle=True,save_model=False):

        # shuffle : True for item2vec, False for word2vec
        # sg: 1 for skip-grams 0 for cbow
        # size: dimension of embedding
        # save_modle: True for save model


         self.sentences = sentences
         self.epochs = epochs
         self.size = size
         self.sg = sg
         self.n_jobs = n_jobs
         self.shuffle = shuffle
         self.save_model = save_model
         self.embedding_lookup = self.train()

    def train(self):
        if self.shuffle:
            start = datetime.now()
            model = Word2Vec(sentences=self.sentences,
                             iter=1,
                             min_count=1,
                             size=self.size,
                             workers=self.n_jobs,
                             sg=self.sg,
                             hs=0,
                             negative=5,

                             window=5)
            print('epoch 1 time: {}'.format(datetime.now() - start))

            for i in range(self.epochs - 1):
                # shuffle sentences
                for sent in self.sentences:
                    random.shuffle(sent)
                start = datetime.now()
                model.train(self.sentences, total_examples=len(self.sentences), epochs=1)
                print('epoch {} time: {}'.format(i + 2, datetime.now() - start))
                if self.save_model and i ==self.epochs-2 :
                    model.save('item2vec.model')
        else:
            start = datetime.now()
            model = Word2Vec(sentences=self.sentences,
                             iter=self.epochs,
                             min_count=1,
                             size=self.size,
                             workers=self.n_jobs,
                             sg=self.sg,
                             hs=0,
                             negative=5,
                             window=5)
            print('Training Time: {}'.format(datetime.now() - start))
            if self.save_model:
                model.save('word2vec.model')
        return model




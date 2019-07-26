import pandas as pd
import numpy as np
from tqdm import tqdm


def partition_num(num, workers):

    if num % workers == 0:

        return [num // workers] * workers

    else:

        return [num // workers] * workers + [num % workers]


def lookup2df(lookup,words_list,size,columns = None,id2word=None):

    # size: embedding dimension
    # columns: set the column names for dataframe
    # id2word: transform the index from ids to words themselves

    tmp_dict = {}
    for index in tqdm(set(words_list)):
        tmp_dict[index] = lookup[str(index)]
    tmp_df = pd.DataFrame()

    tmp_df['wordid'] = tmp_dict.keys()
    for i in range(size):
        tmp_df['word_embedding_' + str(i)] = tmp_df['wordid'].apply(lambda x: tmp_dict[x][i])
        print('Embedding ', i, ' Done')

    if columns:
        tmp_df.columns = columns

    if id2word:
        tmp_df['wordid'] = tmp_df['wordid'].apply(lambda x: id2word[x])

    return tmp_df
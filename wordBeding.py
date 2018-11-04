import pandas as pd
import numpy as np

import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
from gensim.test.utils import get_tmpfile

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from sklearn.manifold import TSNE

import os
import collections
#import smart_open
import random

import nltk

#import spacy
import time

import multiprocessing

from pprint import pprint
import random


def search_in_doc(similar):
    return [(documents[i[0]], i) for i in similar]


def display_closestwords_tsnescatterplot(model, word):
    
    mpl.style.use('seaborn')
    colors = cm.rainbow(np.linspace(0, 1, 10))
    arr = np.empty((0,300), dtype='f')
    word_labels = [word]
    
    # get close words
    close_words = model.similar_by_word(word)
    
    # add the vector for each of the closest words to the array
    arr = np.append(arr, np.array([model[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)
        
    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    # display scatter plot
    plt.scatter(x_coords, y_coords, color=random.choice(colors))
    
    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
    plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    plt.show()



df = pd.read_csv('Candidatos/Ciro.csv')

test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data'])
lee_train_file = test_data_dir + os.sep + 'lee_background.cor'
lee_test_file = test_data_dir + os.sep + 'lee.cor'

lower_subjects = [subject.lower() for subject in df["text"]]
token_list = [nltk.word_tokenize(s) for s in lower_subjects]
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(token_list)]


cores = multiprocessing.cpu_count()

model = Doc2Vec(
                dm=0,
                dbow_words=1,
                size=300,
                window=10,
                min_count=2,
                iter=10000,
                workers=cores)

fname = get_tmpfile("my_doc2vec_model")

try:
	model = Doc2Vec.load(fname)
except:
	model.build_vocab(documents)

	print("inicio do treino")
	model.train(documents, total_examples=model.corpus_count, epochs=model.iter)
	print("fim do treino")
	model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
	model.save(fname)


phrase = "Ciro"
tokens = nltk.word_tokenize(phrase)

inferred_vector = model.infer_vector(tokens)

similars = model.docvecs.most_similar([inferred_vector], topn=10)

pprint(search_in_doc(similars))

display_closestwords_tsnescatterplot(model, 'ciro')
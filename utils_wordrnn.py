# Supplementary function to load text and create a corpus
import os
import csv
from six.moves import cPickle
import numpy as np
import re
from keras.utils import np_utils
from nltk.tokenize import RegexpTokenizer

def load_corpus(raw_data_address,sequence_length):
    int2word_save_address='wordrnn_int2word.pkl'
    word2int_save_address='wordrnn_word2int.pkl'
    
    raw_data=str(open(raw_data_address).read().lower())
    nltk_tokenizer=RegexpTokenizer(r'\w+') 
    cleaned_corpus=nltk_tokenizer.tokenize(raw_data) # Removes hypenated words as well!
    
    cleaned_corpus_copy=np.copy(cleaned_corpus)
    unique_words=sorted(list(set(cleaned_corpus_copy)))
    len_unique_words=len(unique_words)
    
    int2word=dict((integer,word) for integer,word in enumerate(unique_words))
    word2int=dict((word,integer) for integer,word in enumerate(unique_words))
    
    with open(int2word_save_address,'wb') as f:
        cPickle.dump(int2word,f)
    
    with open(word2int_save_address,'wb') as f:
        cPickle.dump(word2int,f)
        
    X_temp=[]
    Y_temp=[]
    
        
    for i in range(0,len(cleaned_corpus)-sequence_length):
        seq_in=cleaned_corpus[i:i+sequence_length]
        X_temp.append([word2int[word] for word in seq_in])
        
        seq_out=cleaned_corpus[i+sequence_length]
        Y_temp.append([word2int[seq_out]])
                       
    len_pattern=len(X_temp) # Number of word sequences that we can obtain
    
    X=np.reshape(X_temp,(len_pattern,sequence_length,1)) # Reshape for [samples,timesteps,features]
    X_normalize=X/float(len_unique_words)
    
    Y=np_utils.to_categorical(Y_temp) # One-hot output target
    
    return X_normalize,X_temp,Y,Y_temp,int2word,word2int

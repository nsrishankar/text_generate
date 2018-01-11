# Supplementary function to load text and create a corpus
import os
import csv
from six.moves import cPickle
import numpy as np
import re
from keras.utils import np_utils

def load_corpus(raw_data_address,sequence_length):
    int2char_save_address='int2char.pkl'
    char2int_save_address='char2int.pkl'
    
    raw_data=open(raw_data_address,'r').read().lower()
    raw_data=raw_data[0:1500000] # Using only a fragment of text because of poor training resources.
    # Convoluted cleaning of data
    text = re.sub(r"what's", "", raw_data)
    text = re.sub(r"What's", "", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"I'm", "I am", text)
    text = re.sub(r" m ", " am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    #text=re.sub("[^a-zA-Z]"," ",text) # Remove all punctuation and numbering
    text=re.sub("[^a-zA-Z.'?,-]"," ",text) # Still need some aspect of punctuation
    characters=sorted(list(set(text)))
    
    len_data=len(text)
    len_vocabulary=len(characters)
    
    int2char=dict((integer,char) for integer,char in enumerate(characters))
    char2int=dict((char,integer) for integer,char in enumerate(characters))
    
    with open(int2char_save_address,'wb') as f:
        cPickle.dump(int2char,f)
    
    with open(char2int_save_address,'wb') as f:
        cPickle.dump(char2int,f)
        
    X_temp=[]
    Y_temp=[]
    
    for i in range(len_data-sequence_length):
        seq_in=text[i:i+sequence_length]
        X_temp.append([char2int[char] for char in seq_in])
        
        seq_out=text[i+sequence_length]
        Y_temp.append([char2int[seq_out]])
    
    len_pattern=len(X_temp) # Patterns
    
    X=np.reshape(X_temp,(len_pattern,sequence_length,1)) # Reshape for [samples,timesteps,features]
    X_normalize=X/float(len_vocabulary)
    
    Y=np_utils.to_categorical(Y_temp) # One-hot output target
    
    return X_normalize,X_temp,Y,Y_temp,len_data,len_vocabulary,int2char,char2int
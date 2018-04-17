
# coding: utf-8

# # Text Generation
# 
# ## Deep Learning

# For this model, we can decide between either GRU or LSTM units. GRU units train faster and are simpler than LSTMS. However, since this project has longer sequences and require longer relationship modeling, an LSTM unit was chosen. This was a fun project inspired by Understanding LSTMs (Colah's blog) and The Unreasonable Effectiveness of Recurrent Neural Networks (Andrej Karpathy's blog). 
# 
# Char-RNN that predicts character sequences was done and generated some HP texts but also some nonsensical words. This could mean that more training is needed, or a better algorithm is needed. Hence, this one is a word-RNN to predict word sequences instead.
# 
# Additionally, I have used smaller RNN networks (not as many LSTM-Dropout repetitions) as they are faster to train (even though it might not be the *best* network).

# ---
# 
# ## Step 0: Import required packages

# In[1]:

## LIST OF ALL IMPORTS
print("Into Script")
import argparse
import os
import math
import random
import time
import re
import os.path as path
from datetime import datetime
from sys import stdout
import numpy as np
import tensorflow as tf
import keras
from nltk.tokenize import RegexpTokenizer

from keras import Model
from keras.models import load_model, Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, TimeDistributed, SimpleRNN, Activation
from keras.optimizers import RMSprop, Adam, Nadam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import multi_gpu_model
from utils_wordrnn import load_corpus

##############################################################################
# Creating a Multi-GPU wrapper for loading/saving weights
class Mult_GPU_model(Model):
	def __init__(self,raw_model,n_gpu):
		parallel_model=multi_gpu_model(raw_model,n_gpu)
		self.__dict__.update(parallel_model.__dict__)
		self._serial_model=raw_model
	
	def __getattribute__(self,attrname):
		# Override save method from serial model
		if 'save' in attrname:
			return getattr(self._serial_model,attrname)
		return super(Mult_GPU_model,self).__getattribute__(attrname)
#############################################################################
		

parser=argparse.ArgumentParser()
parser.add_argument("-l", "--learning_rate", type=float, help="<Required> Set learning rate", required=True)
parser.parse_args()

args=parser.parse_args()

data_path='raw_hp.txt'
length_sequence=50
print("Loading corpus of texts.")
X_normalize,X_temp,Y,Y_temp,int2word,word2int=load_corpus(data_path,length_sequence) 
# Using a sequence of 1,000,000 raw characters/ not the entire corpus due to memory errors
print("HP Corpus created.")


input_shape=(X_normalize.shape[1],X_normalize.shape[2])
num_layers=0 # Choosing a smaller network for faster training, previous outputs with n_layers==2
rnn_hidden_layers=512
n_epochs=55
batch_size=384
drop_rate=0.05
learning_rate=args.learning_rate
print("LR ",learning_rate)

model=Sequential()
model.add(LSTM(rnn_hidden_layers,input_shape=input_shape,return_sequences=True))
model.add(Dropout(drop_rate))
model.add(LSTM(rnn_hidden_layers))
model.add(Dropout(drop_rate))

for i in range(num_layers):
    if (i==num_layers-1):
        model.add(LSTM(rnn_hidden_layers))
        model.add(Dropout(drop_rate))    
    else:
        model.add(LSTM(rnn_hidden_layers,return_sequences=True))
        model.add(Dropout(drop_rate))
model.add(Dense(Y.shape[1],activation='softmax'))

optimizer=Adam(lr=learning_rate)
model.load_weights("weights_adam_7.5e-05_49-2.0189.hdf5")

parallel_model=Mult_GPU_model(model,n_gpu=3)
parallel_model.compile(loss='categorical_crossentropy',optimizer=optimizer)
parallel_model.summary()

print("\nTraining.")

filepath="weights_adam_"+str(learning_rate)+"_{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
early_stopping=EarlyStopping(monitor='val_loss',min_delta=0.0005,patience=10,mode='min',verbose=1)
callbacks_list = [checkpoint, early_stopping]
    
    
parallel_model.fit(X_normalize,Y,batch_size=batch_size,verbose=1,nb_epoch=n_epochs,callbacks=callbacks_list)
#model.save_weights(filepath)

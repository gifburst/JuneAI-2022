import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, GRU, Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint

data_text = 

import re
def text_cleaner(text):
# lower case text
newString = text.lower()
newString = re.sub(r"'s\b","",newString)
# remove punctuations
newString = re.sub("[^a-zA-Z]", " ", newString) 
long_words=[]
# remove short word
for i in newString.split():
if len(i)>=3:                  
long_words.append(i)
return (" ".join(long_words)).strip()
# preprocess the text
data_new = text_cleaner(data_text)

def create_seq(text):
length = 30
sequences = list()
for i in range(length, len(text)):
# select sequence of tokens
seq = text[i-length:i+1]
# store
sequences.append(seq)
print('Total Sequences: %d' % len(sequences))
return sequences
# create sequences   
sequences = create_seq(data_new)


# create a character mapping index
chars = sorted(list(set(data_new)))
mapping = dict((c, i) for i, c in enumerate(chars))
def encode_seq(seq):
sequences = list()
for line in seq:
# integer encode line
encoded_seq = [mapping[char] for char in line]
# store
sequences.append(encoded_seq)
return sequences
# encode the sequences
sequences = encode_seq(sequences

from sklearn.model_selection import train_test_split
# vocabulary size
vocab = len(mapping)
sequences = np.array(sequences)
# create X and y
X, y = sequences[:,:-1], sequences[:,-1]
# one hot encode y
y = to_categorical(y, num_classes=vocab)
# create train and validation sets
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
print('Train shape:', X_tr.shape, 'Val shape:', X_val.shape)

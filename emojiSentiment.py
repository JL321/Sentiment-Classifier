import numpy as np
import pandas as pd
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model

emojiSet = pd.read_csv("emojify_data.csv", header = None)

emojiSet = emojiSet.iloc[:, [0,1]]
emojiSetA = np.array(emojiSet).tolist()

featureSet = []
labelSet = []

for i, v in enumerate(emojiSetA): #Create featureset
    featureSet.append(v[0])
    labelSet.append(v[1])
    
featureSet = np.array(featureSet)
labelSet = np.array(labelSet)

oneHotLabel = np.array([], dtype=np.int64).reshape(0,5)

for i in labelSet: #Create labels
    new = np.zeros(5)
    new[i] = 1
    oneHotLabel = np.vstack((oneHotLabel,new))

maxval = 0

for string in featureSet:
    string = string.split()
    if len(string) > maxval:
        maxval = len(string)

prep = Tokenizer() #Convert sentences into arrays of strings
prep.fit_on_texts(featureSet)
vocab_length = len(prep.word_index)+1
converted_text = prep.texts_to_sequences(featureSet)
converted_text = pad_sequences(converted_text, maxlen = 10, padding = 'post')

embedding_dict = {}

###Open glove embeddings 
glove = open('glove.6B.100d.txt', encoding="utf8")

for line in glove: #Create the dictionary for glove vectors
    line = line.split()
    embedding_dict[line[0]] = np.asarray(line[1:], dtype = 'float32')

glove.close()

embedding_matrix = np.zeros((vocab_length,100))

for word, i in prep.word_index.items(): #Only use the word embeddings required- only load the words on the training set (loading the 400k+1 vocab on the glove vector takes too long, and is unnecessary)
    embedding_vector = embedding_dict[word]
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

print(embedding_matrix.shape)

from keras.layers import LSTM, Dense, Dropout, Embedding, Input, BatchNormalization

#Start building model

init = Input(shape=(maxval,))
e = Embedding(vocab_length, 100, weights = [embedding_matrix], trainable = False)(init)
x = LSTM(128, return_sequences = True)(e)
x = Dropout(0.8)(x)
x = LSTM(128, return_sequences = True)(x)
x = BatchNormalization()(x)
x = Dropout(0.8)(x)
x = LSTM(128)(x)
x = Dense(5, activation='softmax')(x)

model = Model(inputs = init, outputs = x)
model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(converted_text, oneHotLabel, validation_split = .2, epochs = 50, verbose = 2)

model.save('sentimentClassifier.h5')


new_model = load_model('sentimentClassifier.h5')
new_input = input('Enter a sentence:')
new_input = new_input.lower()
new_input = new_input.split()

ignore = True

for i, word in enumerate(new_input): #If an unknown word exists, then set that to 0
    for unk in prep.word_index:
        if unk == word:
            #print(word)
            ignore = False
    if ignore:
        new_input[i] = 0
    else:
        new_input[i] = prep.word_index[word]
    ignore = True

counter = len(new_input)

while counter < maxval:
    new_input.append(0) 
    counter += 1

new_input1 =  np.array([], dtype=np.int64).reshape(0,10)
new_input = np.array(new_input)
new_input1 = np.vstack((new_input1,new_input))
new_input1 = np.vstack((new_input1,np.array([0,0,0,0,0,0,0,0,0,0]))) #Temporary fix - currently requires 2 array input for prediction to run

store = new_model.predict(new_input1)

store = np.argmax(store[0]) #Find Max Val

if store == 0: #Print Emojis
    print('\U0001F60D'+'\U0001F60A')
elif store == 1:
    print('\U000026BE')
elif store == 2:
    print('\U0001F600'+'\U0001F603')
elif store == 3:
    print('\U0001F620'+'\U0001F614')
elif store == 4:
    print('\U0001F357'+'\U0001F96A	')
                	

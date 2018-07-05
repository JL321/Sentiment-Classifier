import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import rnn
import re
import datetime

tf.reset_default_graph()

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

int_to_string = {}
string_to_int = {}

maxval = 0

for i,string in enumerate(featureSet):
    string = string.lower()
    sentence = string.split()
    for a,v in enumerate(sentence):
        if v == "i'm":
            sentence.insert(a+1, "i")
            sentence.insert(a+2, "am")
            del sentence[a]
        if v == "don't":
            sentence.insert(a+1, "do")
            sentence.insert(a+2, "not")
            del sentence[a]
    if len(sentence) > maxval:
        maxval = len(sentence)
    new_s = " ".join(sentence)
    featureSet[i] = new_s

counter = 0

#Creating integer conversion dictionary

for string in featureSet:
    sentence = string.split()
    for word in sentence:
        if word not in string_to_int:
            string_to_int[word] = counter
            int_to_string[counter] = word
            counter += 1
            
word_count = len(string_to_int)

print(string_to_int)
#Convert to one hot encoding

def one_hot(feature):
    blank = np.zeros((word_count))
    pos = string_to_int[feature]
    blank[pos] = 1
    return blank
 
def pad_sequence(feature, maxval):   
    base = np.zeros(((np.array(feature).shape[1]))).tolist()
    while len(feature) < maxval:
        feature.append(base)
    return feature
    
embedded_features = []

embedding_dict = {}

glove = open('glove.6B.100d.txt', encoding="utf8")

for line in glove: #Create the dictionary for glove vectors
    line = line.split()
    embedding_dict[line[0]] = np.asarray(line[1:], dtype = 'float32')

glove.close()

embedding_matrix = np.zeros((word_count,100))

for word in embedding_dict:
    if word in string_to_int:
        for i in range (100):
            embedding_matrix[string_to_int[word]][i] = embedding_dict[word][i]      

for string in featureSet:
    sentence = string.split()
    sent = []
    for i, word in enumerate(sentence):
        sent.append(one_hot(word).tolist())
    sent = pad_sequence(sent, maxval)
    sent = np.dot(np.array(sent), embedding_matrix)
    embedded_features.append(sent)
    
embedded_features = np.array(embedded_features)

batch_size= 16

x = tf.placeholder(tf.float32, [batch_size, maxval, 100])
y = tf.placeholder(tf.int32, [batch_size, 5])

weights = {
        'out': tf.Variable(tf.random_normal([256, 5]))
        }

bias = {
        'out': tf.Variable(tf.random_normal([5]))
        }

def RNN(x, weights, biases):
  
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(256), rnn.BasicLSTMCell(256)])
    
    outputs, state = tf.nn.dynamic_rnn(rnn_cell, x, dtype = tf.float32)
    last_output = outputs[:,-1]
    z1 = tf.matmul(last_output,weights['out'])+bias['out']
    return z1, tf.nn.softmax(z1)

pred, collection = RNN(x, weights, bias)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

correct_pred = tf.equal(tf.argmax(collection,1), tf.argmax(y,1))

accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init = tf.global_variables_initializer()

t_acc = 0
t_loss = 0

saver = tf.train.Saver()

with tf.Session() as sess:
    
    tf.summary.scalar('Loss', cost)
    tf.summary.scalar('Accuracy', accuracy)
    merged = tf.summary.merge_all()
    logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    writer = tf.summary.FileWriter(logdir, sess.graph)

    
    sess.run(init)
    step = 0 
    for i in range (1000):
        batch = []
        b_labels = []
        random_used = np.random.randint(0,200, size= 16)
        for a in random_used:
            batch.append(embedded_features[a])
            b_labels.append(oneHotLabel[a])
        
        _, acc, loss, onehot_pred = sess.run([optimizer, accuracy, cost, pred],
                                             feed_dict = {x: batch, y: b_labels})
        t_acc += acc
        t_loss += loss
        
        print(i)
        
        if (i % 50 == 0):
            summary = sess.run(merged, {x: batch, y: b_labels})
            writer.add_summary(summary, i)
        
        if (i % 100 == 0):
            avg_acc = 100*t_acc/100
            avg_loss = t_loss/100
            print('Average accuracy is ', avg_acc,'. Average loss is ', avg_loss)
            t_acc = 0
            t_loss = 0
            save_path = saver.save(sess, 'models/pretrained_lstm,ckpt', global_step = i)
            
    writer.close()
        
print(__doc__)
import tensorflow as tf
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
# import scipy.io as sc
# import numpy as np
# import xgboost as xgb
# import pywt
# import random
# import pandas as pd
import time
# from sklearn import preprocessing
# from sklearn import svm
import scipy.io as sc
import numpy as np
from scipy.signal import butter, lfilter

#Auto regressive coefficient
def one_hot(y_):
    # Function to encode output labels from number indexes
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    y_ = y_.reshape(len(y_))
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]

len_sample=90
len_a=13500/len_sample
label0=np.zeros(len_a)
label1=np.ones(len_a)
label2=np.ones(len_a)*2
label3=np.ones(len_a)*3
label4=np.ones(len_a)*4
label5=np.ones(len_a)*5
label6=np.ones(len_a)*6
label7=np.ones(len_a)*7
label=np.hstack((label0,label1,label2,label3,label4,label5,label6,label7))
label=np.transpose(label)

print label


from statsmodels.tsa.ar_model import AR
feature = sc.loadmat("/home/xiangzhang/matlabwork/eegmmidb/EEG_ID_label6.mat")
input_data = feature['EEG_ID_label6']

input_data=input_data[:,0:64]# there are 64 features, the last two columns have no use.

data=input_data
print 'data', data.shape
# label=data[:,-1]
data=np.reshape(input_data[:,0:64],[len_a*8,len_sample,64])

full=[]
for j in range(13500*8/90):
    sample=data[j]
    # print sample.shape
    par=[]
    for i in range(64):
        aa=sample[:,i]
        model = AR(aa)
        model_fit = model.fit()
        lag=model_fit.k_ar
        p=model_fit.params
        n_steps=len(p)
        # print len(p)
        par.append(p)
    par=np.array(par)
    par=np.transpose(par)
    # par = preprocessing.scale(par)
    # print par.shape
    full.append(par)


full=np.array(full)

print "full.shape", full.shape
print 'n_steps',n_steps

all=full
# print all.shape
# print label.shape
d=zip(all,label)
np.random.shuffle(d)
all,label= zip(*d)
all=np.array(all)
label=np.array(label)
print label

# n_trainperson=4 # the number of training subjects
n_trainperson_strat=1
ID_testperson=8
n_sample_persubject=len_a
train_data=all[n_sample_persubject*(n_trainperson_strat-1):1050]


test_data=all[1050:n_sample_persubject*8]

column=65
feature_training =train_data[:,0:column-1]
feature_testing =test_data[:,0:column-1]

label_training =label[n_sample_persubject*(n_trainperson_strat-1):1050]
label_testing =label[1125:n_sample_persubject*8]

print label_training.shape
print label_training

# np.random.shuffle(test_data)


label_training =one_hot(label_training)
print label_training.shape

print label_testing
label_testing =one_hot(label_testing)
print label_testing.shape
a=feature_training
b=feature_testing



nodes=64
lameda=0.004
lr=0.005
batch_size=150
train_fea=[]
n_group=7
for i in range(n_group):
    f =a[(0+batch_size*i):(batch_size+batch_size*i)]
    train_fea.append(f)
print (train_fea[0].shape)

train_label=[]
for i in range(n_group):
    f =label_training[(0+batch_size*i):(batch_size+batch_size*i), :]
    train_label.append(f)
print (train_label[0].shape)


# hyperparameters

n_inputs = column-1  # MNIST data input (img shape: 11*99)
# n_steps = 6 # time steps
n_hidden1_units = nodes   # neurons in hidden layer
n_hidden2_units = nodes
n_hidden3_units = nodes
n_hidden4_units=nodes
n_classes = 8

# tf Graph input

x = tf.placeholder(tf.float32, [None,n_steps, n_inputs],name="features")
y = tf.placeholder(tf.float32, [None, n_classes],name="label")

# Define weights

weights = {
    # (28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden1_units]), trainable=True,name="weights_in"),
    #(128,128)
    'hidd2': tf.Variable(tf.random_normal([n_hidden1_units, n_hidden2_units]), trainable=True,name="weights_hidd2"),
    'hidd3': tf.Variable(tf.random_normal([n_hidden2_units, n_hidden3_units]), trainable=True,name="weights_hidd3"),
    'hidd4': tf.Variable(tf.random_normal([n_hidden3_units, n_hidden4_units]), trainable=True,name="weights_hidd4"),
    # (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden4_units, n_classes]), trainable=True,name="weights_out"),

}

biases = {
    # (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden1_units]), name="biases_in"),
    #(128,)
    'hidd2': tf.Variable(tf.constant(0.1, shape=[n_hidden2_units ]), name="biases_hidd2"),
    'hidd3': tf.Variable(tf.constant(0.1, shape=[n_hidden3_units]), name="biases_hidd3"),
    'hidd4': tf.Variable(tf.constant(0.1, shape=[n_hidden4_units]),name="biases_hidd4"),
    # (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes ]), name="biases_out")
}

def RNN(X, weights, biases):
    # hidden layer for input to cell
    ########################################

    X = tf.reshape(X, [-1, n_inputs])

    # into hidden
    X_in = tf.matmul(X, weights['in']) + biases['in']
    X_in = tf.matmul(X_in, weights['hidd2']) + biases['hidd2']
    X_in = tf.matmul(X_in, weights['hidd3']) + biases['hidd3']
    X_in = tf.matmul(X_in, weights['hidd4']) + biases['hidd4']
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden4_units])

    # cell
    ##########################################

    # basic LSTM Cell.
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden4_units, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(n_hidden4_units, forget_bias=1.0, state_is_tuple=True)
    lstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    # lstm cell is divided into two parts (c_state, h_state)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)

    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))    # states is the last outputs
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']

    return results, outputs[-1]



pred,Feature = RNN(x, weights, biases)


l2 = lameda * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())  # L2 loss prevents this overkill neural network to overfit the data
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)) + l2   #sogmoid activation function


train_op = tf.train.AdamOptimizer(lr).minimize(cost)
pred_result =tf.argmax(pred, 1)
label_true =tf.argmax(y, 1)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()
# filename = "/home/xiangzhang/PycharmProjects/untitled/activity_recognition_practice/acc/svm.text"
# f1 = open(filename, 'wb')
# filename = "/home/xiangzhang/PycharmProjects/untitled/activity_recognition_practice/acc/dl.text"
# f2 = open(filename, 'wb')

# f3 = open('/home/xiangzhang/PycharmProjects/untitled/activity_recognition_practice/P.csv', 'wb')
# f4 = open('/home/xiangzhang/PycharmProjects/untitled/activity_recognition_practice/T.csv', 'wb')
with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.Saver()
    step = 0
    start = time.clock()
    while step < 1500:
        for i in range(n_group):
            sess.run(train_op, feed_dict={
                x: train_fea[i],
                y: train_label[i],
                })
        if step % 20 == 0:
            svm_feature_training = sess.run(Feature, feed_dict={
                x: train_fea[i],
                y: train_label[i],
            })
            svm_label_training = np.argmax(train_label[i], 1)

            clf = OneVsRestClassifier(LinearSVC(random_state=0)).fit(svm_feature_training, svm_label_training)
            hh = sess.run(Feature, feed_dict={
                x: b,
                y: label_testing,
            })
            svm_result = clf.predict(hh)
            svm_pred = tf.equal(svm_result, tf.argmax(label_testing, 1))
            acc_svm = sess.run(tf.reduce_mean(tf.cast(svm_pred, tf.float32)))
            print "the setp is:", step, ',learing rate is', lr, 'the svm accuracy is:', acc_svm

            # f1.write(str(acc_svm) + '\n')
            # print("The lamda is :", lamena, ", Learning rate:", lr, ", The step is:", step, ", The accuracy is:", hh)
            acc_dl=sess.run(accuracy, feed_dict={
                x: b,
                y: label_testing,
            })
            print("The dl acc is is :",acc_dl )
            # f2.write(str(acc_dl) + '\n')
        if sess.run(accuracy, feed_dict={
                x: b,
                y: label_testing,
            }) > 0.99:
            print(
                "The lamda is :", lameda, ", Learning rate:", lr, ", The step is:", step, ", The accuracy is: ",
                sess.run(accuracy, feed_dict={
                    x: b,
                    y: label_testing,
                }))
            break
        step += 1
    endtime=time.clock()
    print "run time:", endtime-start
    # f3.write(str(sess.run(pred_result, feed_dict={x: b, y: label_testing})))
    # f4.write(str(sess.run(label_true, feed_dict={x: b, y: label_testing})))
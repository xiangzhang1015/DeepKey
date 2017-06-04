import tensorflow as tf
import scipy.io as sc
import numpy as np
import random

import time
from sklearn import preprocessing

# this function is used to transfer one column label to one hot label
def one_hot(y_):
    # Function to encode output labels from number indexes
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    y_ = y_.reshape(len(y_))
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]


#EEG eegmmidb person dependent raw data mixed read  /home/xiangzhang/scratch/eegmmidb/
##/home/xiangzhang/matlabwork/eegmmidb/
# feature = sc.loadmat("/home/xiangzhang/scratch/eegmmidb/S3_nolabel6.mat")
# all = feature['S3_nolabel6']
#
# print all.shape

# all=all[0:28000]
# feature_all =all[:,0:64]
# label=all[:,64:65]

#z-score
# feature_all=preprocessing.scale(feature_all)
#min-max scaling
# feature_normalized=preprocessing.minmax_scale(feature_all,feature_range=(0,1))

#unity scaling
# feature_normalized=feature_all/sum(feature_all)


#the subject: 1 2   3    9   10      11  12 14  18  6
###########CNN   S9_nolabel6
time1=time.clock()
feature = sc.loadmat("/home/xiangzhang/scratch/eegmmidb/S1_nolabel6.mat")
all = feature['S1_nolabel6']

print (all.shape)

np.random.shuffle(all)   # mix eeg_all

final=2800*10
all=all[0:final]
feature_all =all[:,0:64]
label=all[:,64:65]

#z-score
feature_all=preprocessing.scale(feature_all)
no_fea=feature_all.shape[-1]
# feature_all =feature_all.reshape([28000,1,no_fea])
label_all=one_hot(label)
print (label_all.shape)



###CNN code,
feature_all=feature_all# the input data of CNN
print "cnn input feature shape", feature_all.shape
n_fea=feature_all.shape[-1]

middle_number=final*3/4
feature_training =feature_all[0:middle_number]
feature_testing =feature_all[middle_number:final]
label_training =label_all[0:middle_number]
label_testing =label_all[middle_number:final]
label_ww=label[middle_number:final]##for the confusion matrix
print ("label_testing",label_testing.shape)
a=feature_training
b=feature_testing
print(feature_training.shape)
print(feature_testing.shape)

keep=1
batch_size=final-middle_number
n_group=3
train_fea=[]
for i in range(n_group):
    f =a[(0+batch_size*i):(batch_size+batch_size*i)]
    train_fea.append(f)
print (train_fea[0].shape)

train_label=[]
for i in range(n_group):
    f =label_training[(0+batch_size*i):(batch_size+batch_size*i), :]
    train_label.append(f)
print (train_label[0].shape)

# the CNN code
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess3.run(prediction, feed_dict={xs: v_xs, keep_prob: keep})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess3.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: keep})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# def max_pool_2x2(x):
#     # stride [1, x_movement, y_movement, 1]
#     return tf.nn.max_pool(x, ksize=[1,1,2,1], strides=[1,1,2,1], padding='SAME')
def max_pool_1x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,1,2,1], strides=[1,1,2,1], padding='SAME')

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, n_fea]) # 1*64
ys = tf.placeholder(tf.float32, [None, 6])  # 2 is the classes of the data
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 1, n_fea, 1])
print(x_image.shape)  # [n_samples, 28,28,1]

## conv1 layer ##
W_conv1 = weight_variable([1,1, 1,2]) # patch 1*1, in size is 1, out size is 2
b_conv1 = bias_variable([2])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 1*64*2
h_pool1 = max_pool_1x2(h_conv1)                          # output size 1*32x2

## conv2 layer ##
W_conv2 = weight_variable([1,1, 2, 4]) # patch 1*1, in size 2, out size 4
b_conv2 = bias_variable([4])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 1*32*4
h_pool2 = max_pool_1x2(h_conv2)                          # output size 1*16*4

## fc1 layer ##
W_fc1 = weight_variable([1*(n_fea/4)*4, 120])
b_fc1 = bias_variable([120])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 1*(n_fea/4)*4])
h_fc1 = tf.nn.sigmoid(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
W_fc2 = weight_variable([120, 6])
b_fc2 = bias_variable([6])
prediction = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# the error between prediction and real data
l2 = 0.001 * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=ys))+l2   # Softmax loss
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(tf.clip_by_value(prediction,1e-10,1.0)),reduction_indices=[1]))
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))       # loss  +l2
train_step = tf.train.AdamOptimizer(0.04).minimize(cross_entropy) # learning rate is 0.0001

sess3 = tf.Session()
init = tf.global_variables_initializer()
sess3.run(init)

# for step in range(1000):
#     # batch_xs, batch_ys = mnist.train.next_batch(100)
#     sesss3.run(train_step, feed_dict={xs: train_fea[i], ys: train_label[i], keep_prob: 0.5})
#     if step % 50 == 0:
#         print(compute_accuracy(b,label_testing))
np.set_printoptions(threshold=np.nan)
filename = "/home/xiangzhang/scratch/results/cnn_acc.csv"
f1 = open(filename, 'wb')
step = 1
while step < 2500:
    for i in range(n_group):
        sess3.run(train_step, feed_dict={xs: train_fea[i], ys: train_label[i], keep_prob:keep})
    if step % 5 == 0:
        cost=sess3.run(cross_entropy, feed_dict={xs: b, ys: label_testing, keep_prob: keep})
        acc_cnn_t=compute_accuracy(b, label_testing)
        print('the step is:',step,',the acc is',acc_cnn_t,', the cost is', cost)
        f1.write(str(acc_cnn_t)+'\n')

    step+=1
acc_cnn=compute_accuracy(b, label_testing)
time2=time.clock()
feature_all_cnn=sess3.run(h_fc1_drop, feed_dict={xs: feature_all, keep_prob: keep})
# label_all=label_all # not one-hot, one dimension
# all=np.hstack((feature_all,label_all))
print ("the shape of cnn output features",feature_all.shape,label_all.shape)

time3=time.clock()
print ("CNN train time:", time2-time1, "cnn test time", time3-time2, 'CNN total time', time3-time1)



#AE code

# # n_fea = all.shape[-1] - 1
# # print "AE input feature no.", n_fea
# # feature_all = all[:, 0:66]
# feature_all=feature_all_cnn
# n_fea = feature_all.shape[-1]
# print "AE input feature no.", n_fea
#
# print 'feature all shape',feature_all.shape
#
#
# # label = all[:, n_fea:n_fea + 1]
# train_fea = feature_all[0:21000]
#
# group = 3
# display_step = 10
#
# # Network Parameters
# n_hidden_1 = 200  # 1st layer num features
#
# n_hidden_2 = 100
#
# n_input_ae = n_fea  # MNIST data input (img shape: 28*28)
#
# # tf Graph input (only pictures)
# X = tf.placeholder("float", [None, n_input_ae])
# weights = {
#     'encoder_h1': tf.Variable(tf.random_normal([n_input_ae, n_hidden_1])),
#     'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
#     'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
#     'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input_ae])),
# }
# biases = {
#     'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
#     'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
#     'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
#     'decoder_b2': tf.Variable(tf.random_normal([n_input_ae])),
# }
#
#
# # Building the encoder
# def encoder(x):
#     # Encoder Hidden layer with sigmoid activation #1
#     layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
#                                    biases['encoder_b1']))
#     # Decoder Hidden layer with sigmoid activation #2
#     # layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
#     #                                biases['encoder_b2']))
#     return layer_1
#
#
# # Building the decoder
# def decoder(x):
#     # Encoder Hidden layer with sigmoid activation #1
#     layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h2']),
#                                    biases['decoder_b2']))
#     # Decoder Hidden layer with sigmoid activation #2
#     # layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
#     #                                biases['decoder_b2']))
#     return layer_1
#
#
# for ll in range(1):
#     learning_rate = 0.01 + ll * 0.002
#     for ee in range(1):
#         training_epochs = 100
#         # Construct model
#         encoder_op = encoder(X)
#         decoder_op = decoder(encoder_op)
#         # Prediction
#         y_pred = decoder_op
#         # Targets (Labels) are the input data.
#         y_true = X
#
#         # Define loss and optimizer, minimize the squared error
#         cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
#         # cost = tf.reduce_mean(tf.pow(y_true, y_pred))
#         optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
#
#         # Initializing the variables
#         init = tf.global_variables_initializer()
#
#         # Launch the graph
#         saver = tf.train.Saver()
#         # filename = "/home/xiangzhang/PycharmProjects/untitled/activity_recognition_practice/acc/xgboost_AE_input.csv"
#         # f3 = open(filename, 'wb')
#         # filename = "/home/xiangzhang/PycharmProjects/untitled/activity_recognition_practice/acc/xgboost_AE_middle.csv"
#         # f4 = open(filename, 'wb')
#         with tf.Session() as sess1:
#             sess1.run(init)
#             saver = tf.train.Saver()
#             # Training cycle
#             for epoch in range(training_epochs):
#                 # Loop over all batches
#                 for i in range(group):
#                     # Run optimization op (backprop) and cost op (to get loss value)
#                     _, c = sess1.run([optimizer, cost], feed_dict={X: train_fea})
#                 # Display logs per epoch step
#                 if epoch % display_step == 0:
#                     print("Epoch:", '%04d' % (epoch + 1),
#                           "cost=", "{:.9f}".format(c))
#
#             print("Optimization Finished!")
#
#             ##read the saved AE model, which has the loss as 0.00001
#             # saver.restore(sess1,
#             #               "/home/xiangzhang/PycharmProjects/untitled/activity_recognition_practice/generalmodel2/AE_model781007.ckpt")  # attention: this model should change with different person
#             # # #
#             # save_path = saver.save(sess1, "/home/xiangzhang/PycharmProjects/untitled/activity_recognition_practice/generalmodel2/AE_model" + str(train_number) + str(testID) + str(training_epochs) + str(
#             #     group) + ".ckpt")
#             # print("AE model save to path", save_path)
#             # np.set_printoptions(threshold=np.nan)
#             # f3.write(str(rawdata[0:10000, :]))
#             # Applying encode and decode over test set
#             feature_all_AE = sess1.run(
#                 encoder_op, feed_dict={X: feature_all})
#
#             # all = np.hstack((data_feature_AE, label))
#             # print label
#             # f4.write(str(rawdata[0:10000, :]))
#             # print rawdata[:,-1]
#             print "data_AE shape", all.shape
# time3 = time.clock()



#######RNN
# feature_all=np.hstack((feature_all,feature_all_cnn))
feature_all=feature_all
no_fea=feature_all.shape[-1]
print no_fea
feature_all =feature_all.reshape([final,1,no_fea])
print tf.argmax(label_all,1)


print label_all.shape

# middle_number=21000
feature_training =feature_all[0:middle_number]
feature_testing =feature_all[middle_number:final]
label_training =label_all[0:middle_number]
label_testing =label_all[middle_number:final]
# print "label_testing",label_testing
a=feature_training
b=feature_testing
print(feature_training.shape)
print(feature_testing.shape)
nodes=64
lameda=0.004
lr=0.005

batch_size=final-middle_number
train_fea=[]
n_group=3
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

n_inputs = no_fea  # MNIST data input (img shape: 11*99)
n_steps = 1 # time steps
n_hidden1_units = nodes   # neurons in hidden layer
n_hidden2_units = nodes
n_hidden3_units = nodes
n_hidden4_units=nodes
n_classes = 6      # MNIST classes (0-9 digits)

# tf Graph input

x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
# x = tf.placeholder(tf.float32, [None, n_steps, n_inputs], name="features")
y = tf.placeholder(tf.float32, [None, n_classes])

# Define weights

weights = {
# (28, 128)
'in': tf.Variable(tf.random_normal([n_inputs, n_hidden1_units]), trainable=True),
'a': tf.Variable(tf.random_normal([n_hidden1_units, n_hidden1_units]), trainable=True),
#(128,128)
'hidd2': tf.Variable(tf.random_normal([n_hidden1_units, n_hidden2_units])),
'hidd3': tf.Variable(tf.random_normal([n_hidden2_units, n_hidden3_units])),
'hidd4': tf.Variable(tf.random_normal([n_hidden3_units, n_hidden4_units])),
# (128, 10)
'out': tf.Variable(tf.random_normal([n_hidden4_units, n_classes]), trainable=True),
}

biases = {
# (128, )
'in': tf.Variable(tf.constant(0.1, shape=[n_hidden1_units])),
#(128,)
'hidd2': tf.Variable(tf.constant(0.1, shape=[n_hidden2_units ])),
'hidd3': tf.Variable(tf.constant(0.1, shape=[n_hidden3_units])),
'hidd4': tf.Variable(tf.constant(0.1, shape=[n_hidden4_units])),
# (10, )
'out': tf.Variable(tf.constant(0.1, shape=[n_classes ]), trainable=True)
}


def RNN(X, weights, biases):
    # hidden layer for input to cell
    ########################################

    # transpose the inputs shape from
    # X ==> (128 batch * 28 steps, 28 inputs)
    X = tf.reshape(X, [-1, n_inputs])

    # into hidden
    # X_in = (128 batch * 28 steps, 128 hidden)
    X_hidd1 = tf.matmul(X, weights['in']) + biases['in']
    X_hidd2 = tf.matmul(X_hidd1, weights['hidd2']) + biases['hidd2']
    X_hidd3 = tf.matmul(X_hidd2, weights['hidd3']) + biases['hidd3']
    X_hidd4 = tf.matmul(X_hidd3, weights['hidd4']) + biases['hidd4']
    # X_in ==> (128 batch, 28 steps, 128 hidden)
    X_in = tf.reshape(X_hidd4, [-1, n_steps, n_hidden4_units])


    # cell
    ##########################################

    # basic LSTM Cell.
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden4_units, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(n_hidden4_units, forget_bias=1.0, state_is_tuple=True)
    lstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    # lstm cell is divided into two parts (c_state, h_state)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    # You have 2 options for following step.
    # 1: tf.nn.rnn(cell, inputs);
    # 2: tf.nn.dynamic_rnn(cell, inputs).
    # If use option 1, you have to modified the shape of X_in, go and check out this:
    # https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
    # In here, we go for option 2.
    # dynamic_rnn receive Tensor (batch, steps, inputs) or (steps, batch, inputs) as X_in.
    # Make sure the time_major is changed accordingly.
    with tf.variable_scope('lstm1'):
        outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)

    # hidden layer for output as the final results
    #############################################
    # results = tf.matmul(final_state[1], weights['out']) + biases['out']

    # # or
    # unpack to list [(batch, outputs)..] * steps
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))    # states is the last outputs
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']

    return results, outputs[-1]



pred,Feature = RNN(x, weights, biases)
lamena =lameda
l2 = lamena * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())  # L2 loss prevents this overkill neural network to overfit the data
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)) + l2  # Softmax loss
train_op = tf.train.AdamOptimizer(lr).minimize(cost)
    # train_op = tf.train.AdagradOptimizer(l).minimize(cost)
    # train_op = tf.train.RMSPropOptimizer(0.00001).minimize(cost)
    # train_op = tf.train.AdagradDAOptimizer(0.01).minimize(cost)
    # train_op = tf.train.GradientDescentOptimizer(0.00001).minimize(cost)
# pred_result =tf.argmax(pred, 1)
label_true =tf.argmax(y, 1)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
confusion_m=tf.confusion_matrix(tf.argmax(y, 1), tf.argmax(pred, 1))
with tf.Session() as sess:
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()
    step = 0
    filename = "/home/xiangzhang/scratch/results/rnn_acc.csv"
    f2 = open(filename, 'wb')
    while step < 2500:
        for i in range(n_group):
            sess.run(train_op, feed_dict={
                x: train_fea[i],
                y: train_label[i],
            })
        # sess.run(train_op, feed_dict={
        #     x: train_fea[0],
        #     y: train_label[0],
        # })
        if sess.run(accuracy, feed_dict={x: b,y: label_testing,})>0.993:
            print(
            "The lamda is :", lamena, ", Learning rate:", lr, ", The step is:", step, ", The accuracy is: ",
            sess.run(accuracy, feed_dict={
                x: b,
                y: label_testing,
            }))

            break
        if step % 5 == 0:
            hh=sess.run(accuracy, feed_dict={
                x: b,
                y: label_testing,
            })
            f2.write(str(hh)+'\n')
            #  "The lamda is :",lamena,", Learning rate:",lr,
            print(", The step is:",step,", The accuracy is:", hh, "The cost is :",sess.run(cost, feed_dict={
                x: b,
                y: label_testing,
            }))

            # np.set_printoptions(threshold='nan') # output all the values, without the apostrophe

            # print()
        step += 1

    ##confusion matrix
    time4 = time.clock()
    feature_0=sess.run(Feature, feed_dict={x: train_fea[0]})
    for i in range(1,n_group):
        feature_11=sess.run(Feature, feed_dict={x: train_fea[i]})
        feature_0=np.vstack((feature_0,feature_11))

    print feature_0.shape
    feature_b = sess.run(Feature, feed_dict={x: b})
    feature_all_rnn=np.vstack((feature_0,feature_b))

    confusion_m=sess.run(confusion_m, feed_dict={
                x: b,
                y: label_testing,
            })
    print confusion_m
    time5 = time.clock()
    ## predict probility
    # pred_prob=sess.run(pred, feed_dict={
    #             x: b,
    #             y: label_testing,
    #         })
    # # print pred_prob


    ##true label is label_true
    # f1.write(str(sess.run(pred_result, feed_dict={x: b, y: label_testing})))
    # f2.write(str(sess.run(label_true, feed_dict={x: b, y: label_testing})))
    # f3.write(str(pred_prob))
    # print 'acc of CNN',acc_cnn

    #
    # print ("CNN run time:", time2 - time1)
    # print "AE time:", time3 - time2
    # print "RNN time",time4-time3
    print ("RNN train time:", time4 - time3, "Rnn test time", time5 - time4, 'RNN total time', time5 - time3)
    # save_path = saver.save(sess, "generalmodel2/eeg_rawdata_runn_model" +  str(lamena) + str(
    #     lr) +str(nodes)+str(n_group)+ ".ckpt")
    # print("save to path", save_path)





###second RNN
print feature_all_rnn.shape, feature_all_cnn.shape
feature_all=np.hstack((feature_all_rnn,feature_all_cnn))
no_fea=feature_all.shape[-1]

# feature_all =feature_all.reshape([28000,1,no_fea])
print label_all.shape

# middle_number=21000
feature_training =feature_all[0:middle_number]
feature_testing =feature_all[middle_number:final]
label_training =label_all[0:middle_number]
label_testing =label_all[middle_number:final]
# print "label_testing",label_testing
a=feature_training
b=feature_testing

##AE

feature_all=feature_all


train_fea=feature_all[0:middle_number]


group=3
display_step = 10
training_epochs = 400

# Network Parameters
n_hidden_1 = 800# 1st layer num features, should be times of 8


n_hidden_2=100

n_input_ae = no_fea # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input_ae])
weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input_ae, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input_ae])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input_ae])),
}


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    # layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
    #                                biases['encoder_b2']))
    return layer_1


# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h2']),
                                   biases['decoder_b2']))
    # Decoder Hidden layer with sigmoid activation #2
    # layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
    #                                biases['decoder_b2']))
    return layer_1

for ll in range(1):
    learning_rate = 0.2
    for ee in range(1):
        # Construct model
        encoder_op = encoder(X)
        decoder_op = decoder(encoder_op)
        # Prediction
        y_pred = decoder_op
        # Targets (Labels) are the input data.
        y_true = X

        # Define loss and optimizer, minimize the squared error
        cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
        # cost = tf.reduce_mean(tf.pow(y_true, y_pred))
        optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

        # Initializing the variables
        init = tf.global_variables_initializer()

        # Launch the graph
        saver = tf.train.Saver()
        # filename = "/home/xiangzhang/PycharmProjects/untitled/activity_recognition_practice/acc/xgboost_AE_input.csv"
        # f3 = open(filename, 'wb')
        # filename = "/home/xiangzhang/PycharmProjects/untitled/activity_recognition_practice/acc/xgboost_AE_middle.csv"
        # f4 = open(filename, 'wb')
        with tf.Session() as sess1:
            sess1.run(init)
            saver = tf.train.Saver()
            # Training cycle
            for epoch in range(training_epochs):
                # Loop over all batches
                for i in range(group):
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = sess1.run([optimizer, cost], feed_dict={X: a})
                # Display logs per epoch step
                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch+1),
                          "cost=", "{:.9f}".format(c))

            print("Optimization Finished!")
            time6=time.clock()
            a = sess1.run(encoder_op, feed_dict={X: a})
            b = sess1.run(encoder_op, feed_dict={X: b})

time7=time.clock()
print ("AE train time:", time6 - time5, "AE test time", time7 - time6, 'AE total time', time7 - time5)


###SVM  RF  KNN
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.svm import LinearSVC
# clf = OneVsRestClassifier(LinearSVC(random_state=0)).fit(a, label_training)
# time2 = time.clock()
# svm_result = clf.score(b, label_testing)
# print svm_result
# time3 = time.clock()
# print "training time", time2 - time1
# print "testing time", time3 - time2
#
# #RF
# from sklearn.ensemble import RandomForestClassifier
#
# rf=RandomForestClassifier(n_estimators=500).fit(a, label_training)
# time2=time.clock()
# rf_result = rf.score(b,label_testing)
# print rf_result
# time3=time.clock()
# print "training time",time2-time1
# print "testing time", time3-time2
#
# ##KNN
# from sklearn.neighbors import KNeighborsClassifier
# for i in range(2,4):
#     neigh = KNeighborsClassifier(n_neighbors=5*i)
#     neigh.fit(a, label_training)
#     time2 = time.clock()
#     knn_result = neigh.score(b, label_testing)
#     print knn_result
#     time3 = time.clock()
#     print "training time", time2 - time1
#     print "testing time", time3 - time2

##XGBoost
import xgboost as xgb
xg_train = xgb.DMatrix(a, label=np.argmax(label_training,1))
xg_test = xgb.DMatrix(b, label=np.argmax(label_testing,1))

# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softprob' # can I replace softmax by SVM??
# softprob produce a matrix with probability value of each class
# scale weight of positive examples
param['eta'] = 0.5

param['max_depth'] = 6
param['silent'] = 1
param['nthread'] = 4
param['subsample']=0.9
# param['lambda']=1
param['num_class'] =6

filename = "/home/xiangzhang/scratch/results/xgb_prob.csv"
f3 = open(filename, 'wb')


np.set_printoptions(threshold=np.nan)
watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
num_round = 500
bst = xgb.train(param, xg_train, num_round, watchlist );
time8=time.clock()
pred = bst.predict( xg_test );
#
# print ('predicting, classification error=%f' % (sum( int(pred[i]) != label_testing[i] for i in range(len(label_testing))) / float(len(label_testing)) ))
time9=time.clock()
f3.write(str(pred)+'\n') #the probability


print ("CNN train time:", time2-time1, "cnn test time", time3-time2, 'CNN total time', time3-time1)
print ("RNN train time:", time4 - time3, "Rnn test time", time5 - time4, 'RNN total time', time5 - time3)
print ("AE train time:", time6 - time5, "AE test time", time7 - time6, 'AE total time', time7 - time5)
print ("XGB train time:", time8 - time7, "XGB test time", time9 - time8, 'XGB total time', time9 - time7)
print 'total train time', time2-time1+time4 - time3+time6 - time5+time8 - time7, 'total test time',time3-time2+time5 - time4+time7 - time6+time9 - time8, 'total run time', time9-time1


# filename = "/home/xiangzhang/scratch/results/xgb_T.csv"
# f4 = open(filename, 'wb')
#
# filename = "/home/xiangzhang/scratch/results/xgb_P.csv"
# f5 = open(filename, 'wb')
# #
# from sklearn.metrics import classification_report
# # print pred.shape, label_testing.shape
# label_ww = [int(x) for x in label_ww]
# pred = [int(x) for x in pred]
# print(classification_report(label_ww, pred))
# f4.write(str(label_ww))
# f5.write(str(pred))
# #
# from sklearn.metrics import confusion_matrix
#
# print  confusion_matrix(label_ww, pred)
##RNN
# no_fea=a.shape[-1]
# a =a.reshape([21000,1,no_fea])
# b =b.reshape([7000,1,no_fea])
# # print(feature_training.shape)
# # print(feature_testing.shape)
# nodes=64
# lameda=0.004
# lr=0.002
#
# batch_size=7000
# train_fea=[]
# n_group=3
# for i in range(n_group):
#     f =a[(0+batch_size*i):(batch_size+batch_size*i)]
#     train_fea.append(f)
# print (train_fea[0].shape)
#
# train_label=[]
# for i in range(n_group):
#     f =label_training[(0+batch_size*i):(batch_size+batch_size*i), :]
#     train_label.append(f)
# print (train_label[0].shape)
#
#
# # hyperparameters
#
# n_inputs = no_fea  # MNIST data input (img shape: 11*99)
# n_steps = 1 # time steps
# n_hidden1_units = nodes   # neurons in hidden layer
# n_hidden2_units = nodes
# n_hidden3_units = nodes
# n_hidden4_units=nodes
# n_classes = 6      # MNIST classes (0-9 digits)
#
# # tf Graph input
#
# x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
# # x = tf.placeholder(tf.float32, [None, n_steps, n_inputs], name="features")
# y = tf.placeholder(tf.float32, [None, n_classes])
#
# # Define weights
#
# weights = {
# # (28, 128)
# 'in2': tf.Variable(tf.random_normal([n_inputs, n_hidden1_units]), trainable=True),
# 'a2': tf.Variable(tf.random_normal([n_hidden1_units, n_hidden1_units]), trainable=True),
# #(128,128)
# 'hidd22': tf.Variable(tf.random_normal([n_hidden1_units, n_hidden2_units])),
# 'hidd32': tf.Variable(tf.random_normal([n_hidden2_units, n_hidden3_units])),
# 'hidd42': tf.Variable(tf.random_normal([n_hidden3_units, n_hidden4_units])),
# # (128, 10)
# 'out2': tf.Variable(tf.random_normal([n_hidden4_units, n_classes]), trainable=True),
# }
#
# biases = {
# # (128, )
# 'in2': tf.Variable(tf.constant(0.1, shape=[n_hidden1_units])),
# #(128,)
# 'hidd22': tf.Variable(tf.constant(0.1, shape=[n_hidden2_units ])),
# 'hidd32': tf.Variable(tf.constant(0.1, shape=[n_hidden3_units])),
# 'hidd42': tf.Variable(tf.constant(0.1, shape=[n_hidden4_units])),
# # (10, )
# 'out2': tf.Variable(tf.constant(0.1, shape=[n_classes ]), trainable=True)
# }
#
#
# def RNN(X, weights, biases):
#     # hidden layer for input to cell
#     ########################################
#
#     # transpose the inputs shape from
#     # X ==> (128 batch * 28 steps, 28 inputs)
#     X = tf.reshape(X, [-1, n_inputs])
#
#     # into hidden
#     # X_in = (128 batch * 28 steps, 128 hidden)
#     X_hidd1 = tf.matmul(X, weights['in2']) + biases['in2']
#     X_hidd2 = tf.matmul(X_hidd1, weights['hidd22']) + biases['hidd22']
#     X_hidd3 = tf.matmul(X_hidd2, weights['hidd32']) + biases['hidd32']
#     X_hidd4 = tf.matmul(X_hidd3, weights['hidd42']) + biases['hidd42']
#     # X_in ==> (128 batch, 28 steps, 128 hidden)
#     X_in = tf.reshape(X_hidd4, [-1, n_steps, n_hidden4_units])
#
#
#     # cell
#     ##########################################
#
#     # basic LSTM Cell.
#     lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden4_units, forget_bias=1.0, state_is_tuple=True)
#     lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(n_hidden4_units, forget_bias=1.0, state_is_tuple=True)
#     lstm_cell2 = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
#     # lstm cell is divided into two parts (c_state, h_state)
#     init_state = lstm_cell2.zero_state(batch_size, dtype=tf.float32)
#
#     # You have 2 options for following step.
#     # 1: tf.nn.rnn(cell, inputs);
#     # 2: tf.nn.dynamic_rnn(cell, inputs).
#     # If use option 1, you have to modified the shape of X_in, go and check out this:
#     # https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
#     # In here, we go for option 2.
#     # dynamic_rnn receive Tensor (batch, steps, inputs) or (steps, batch, inputs) as X_in.
#     # Make sure the time_major is changed accordingly.
#     with tf.variable_scope('lstm2'):
#         outputs, final_state = tf.nn.dynamic_rnn(lstm_cell2, X_in, initial_state=init_state, time_major=False)
#
#     # hidden layer for output as the final results
#     #############################################
#     # results = tf.matmul(final_state[1], weights['out']) + biases['out']
#
#     # # or
#     # unpack to list [(batch, outputs)..] * steps
#     outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))    # states is the last outputs
#     results = tf.nn.softmax(tf.matmul(outputs[-1], weights['out2']) + biases['out2'])
#
#     return results, outputs[-1]
#
#
#
# pred,Feature = RNN(x, weights, biases)
# lamena =lameda
# l2 = lamena * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())  # L2 loss prevents this overkill neural network to overfit the data
# print 'pred.shape',pred[0]
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)) + l2  # Softmax loss
# train_op = tf.train.AdamOptimizer(lr).minimize(cost)
#     # train_op = tf.train.AdagradOptimizer(l).minimize(cost)
#     # train_op = tf.train.RMSPropOptimizer(0.00001).minimize(cost)
#     # train_op = tf.train.AdagradDAOptimizer(0.01).minimize(cost)
#     # train_op = tf.train.GradientDescentOptimizer(0.00001).minimize(cost)
# # pred_result =tf.argmax(pred, 1)
# label_true =tf.argmax(y, 1)
# correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# confusion_m=tf.confusion_matrix(tf.argmax(y, 1), tf.argmax(pred, 1))
# with tf.Session() as sess5:
#     if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
#         init = tf.initialize_all_variables()
#     else:
#         init = tf.global_variables_initializer()
#     sess5.run(init)
#     saver = tf.train.Saver()
#     step = 0
#     # f1 = open('/home/xiangzhang/PycharmProjects/untitled/activity_recognition_practice/layer1.csv', 'wb')
#     # f2 = open('/home/xiangzhang/scratch/acc/layer_true.csv', 'wb')
#     # f3 = open('/home/xiangzhang/scratch/acc/pred_prob.csv', 'wb')
#     # filename = "/home/xiangzhang/PycharmProjects/untitled/activity_recognition_practice/acc/1.text"
#     # f = open(filename, 'a')
#     while step < 4500:
#
#         sess5.run(train_op, feed_dict={
#             x: train_fea[0],
#             y: train_label[0],
#             })
#         if sess5.run(accuracy, feed_dict={x: b,y: label_testing,})>0.993:
#             print(
#             "The lamda is :", lamena, ", Learning rate:", lr, ", The step is:", step, ", The accuracy is: ",
#             sess5.run(accuracy, feed_dict={
#                 x: b,
#                 y: label_testing,
#             }))
#
#             break
#         if step % 20 == 0:
#             hh=sess5.run(accuracy, feed_dict={
#                 x: b,
#                 y: label_testing,
#             })
#             #  "The lamda is :",lamena,", Learning rate:",lr,
#             print(", The step is:",step,", The accuracy is:", hh, "The cost is :",sess5.run(cost, feed_dict={
#                 x: b,
#                 y: label_testing,
#             }))
#
#         step += 1
#
#     ##confusion matrix
#     confusion_m=sess5.run(confusion_m, feed_dict={
#                 x: b,
#                 y: label_testing,
#             })
#     print confusion_m
#     time4 = time.clock()
#     ## predict probility
#     time5=time.clock()
#
#     ##true label is label_true
#     # f1.write(str(sess.run(pred_result, feed_dict={x: b, y: label_testing})))
#     # f2.write(str(sess.run(label_true, feed_dict={x: b, y: label_testing})))
#     # f3.write(str(pred_prob))
#     # print 'acc of CNN',acc_cnn
#
#     #
#     # print ("CNN run time:", time2 - time1)
#     # print "AE time:", time3 - time2
#     # print "RNN time",time4-time3
#     print "training time:", time4-time1
#     print "testing time:", time5 - time4



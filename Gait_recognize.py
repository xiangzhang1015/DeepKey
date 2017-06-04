import tensorflow as tf
import scipy.io as sc
import numpy as np
import random

# this function is used to transfer one column label to one hot label
def one_hot(y_):
    # Function to encode output labels from number indexes
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    y_ = y_.reshape(len(y_))
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]


#AR mixed raw data read
# Walking_all is the normalized data of AR_ID_8 person.
# #It seems that the normalized data has better performance here and the raw data works better in invalid_filter.py.
all=sc.loadmat("/home/xiangzhang/matlabwork/personrecognize/walking_all.mat")
all=all["walking_all"]
# feature = sc.loadmat("/home/xiangzhang/matlabwork/AR_ID_8person.mat")
# all = feature['AR_ID_8person']
np.random.shuffle(all)
all_training=all[0:140000,:]
all_testing=all[140000:160000,:]

feature_training =all_training[:,0:51]
feature_testing =all_testing[:,0:51]
label_training =all_training[:,51:52]
label_testing =all_testing[:,51:52]
print label_testing.shape

feature_training=np.reshape(feature_training,[140000,1,51])
feature_testing=np.reshape(feature_testing,[20000,1,51])
label_training=one_hot(label_training)
label_testing=one_hot(label_testing)
print feature_training.shape
print feature_testing.shape
print label_training.shape
print label_testing.shape




a=feature_training
b=feature_testing


#batch split
batch_size=20000
train_fea=[]
n_group=7
for i in range(n_group):
    f =a[(0+batch_size*i):(batch_size+batch_size*i), :, :]
    train_fea.append(f)
print train_fea[0].shape

train_label=[]
for i in range(n_group):
    f =label_training[(0+batch_size*i):(batch_size+batch_size*i), :]
    train_label.append(f)
print (train_label[0].shape)


# hyperparameters
n_inputs = 51 # MNIST data input (img shape: 11*99)
n_steps = 1  # time steps
n_hidden1_units = 32   # neurons in hidden layer
n_hidden2_units = 32
n_hidden3_units = 32
n_hidden4_units = 32

n_classes = 9     # 8 person add label 0

# tf Graph input
with tf.name_scope('inputs'):
    x = tf.placeholder(tf.float32, [None, n_steps, n_inputs],name="features")
    y = tf.placeholder(tf.float32, [None, n_classes],name="label")

    # Define weights
    with tf.name_scope('weights'):
        weights = {
            # (28, 128)
            'in': tf.Variable(tf.random_normal([n_inputs, n_hidden1_units]), name="weights_in"),
            #(128,128)
            'hidd2': tf.Variable(tf.random_normal([n_hidden1_units, n_hidden2_units]), name="weights_hidd2"),
            'hidd3': tf.Variable(tf.random_normal([n_hidden2_units, n_hidden3_units]), name="weights_hidd3"),
            'hidd4': tf.Variable(tf.random_normal([n_hidden3_units, n_hidden4_units]), name="weights_hidd4"),
            # (128, 10)
            'out': tf.Variable(tf.random_normal([n_hidden3_units, n_classes]), name="weights_out"),

        }
    with tf.name_scope('biases'):
        biases = {
            # (128, )
            'in': tf.Variable(tf.constant(0.1, shape=[n_hidden1_units]), name="biases_in"),
            #(128,)
            'hidd2': tf.Variable(tf.constant(0.1, shape=[n_hidden2_units ]), name="biases_hidd2"),
            'hidd3': tf.Variable(tf.constant(0.1, shape=[n_hidden3_units]), name="biases_hidd3"),
            'hidd4': tf.Variable(tf.constant(0.1, shape=[n_hidden4_units]), name="biases_hidd4"),
            # (10, )
            'out': tf.Variable(tf.constant(0.1, shape=[n_classes ]), name="biases_out")
        }
def RNN(X, weights, biases):
    # hidden layer for input to cell
    ########################################

    # transpose the inputs shape from

    X = tf.reshape(X, [-1, n_inputs])

    # into hidden
    X_in = tf.matmul(X, weights['in']) + biases['in']
    X_in = tf.matmul(X_in, weights['hidd2']) + biases['hidd2']
    X_in = tf.matmul(X_in, weights['hidd3']) + biases['hidd3']
    X_in = tf.matmul(X_in, weights['hidd4']) + biases['hidd4']
    # X_in ==> (128 batch, 28 steps, 128 hidden)
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

    return results



pred = RNN(x, weights, biases)
lamena =0.001
l2 = lamena * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())  # L2 loss prevents this overkill neural network to overfit the data
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=pred)) + l2  # Softmax loss

lr=0.001
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
#
# f3 = open('/home/xiangzhang/PycharmProjects/untitled/activity_recognition_practice/P_AR.csv', 'wb')
# f4 = open('/home/xiangzhang/PycharmProjects/untitled/activity_recognition_practice/T_AR.csv', 'wb')
with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step < 1000:
        for i in range(n_group):
            sess.run(train_op, feed_dict={
                x: train_fea[i],
                y: train_label[i],
                })
        if step % 10 == 0:
            print("The lamda is :",lamena,", Learning rate:",lr,", The step is:",step,", The accuracy is: ",sess.run(accuracy, feed_dict={
                x: b,
                y: label_testing,
            }))
            np.set_printoptions(threshold='nan') # output all the values, without the apostrophe
            print("The cost is :",sess.run(cost, feed_dict={
                x: b,
                y: label_testing,
            }))
        if sess.run(accuracy, feed_dict={x: b, y: label_testing, }) > 0.999:
            print(
                "The lamda is :", lamena, ", Learning rate:", lr, ", The step is:", step,
                ", The accuracy is: ",
                sess.run(accuracy, feed_dict={
                    x: b,
                    y: label_testing,
                }))

            break
        step += 1
    # f3.write(str(sess.run(pred_result, feed_dict={x: b, y: label_testing})))
    # f4.write(str(sess.run(label_true, feed_dict={x: b, y: label_testing})))




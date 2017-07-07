# -*- coding: utf-8 -*-
import sys, glob
reload(sys)
sys.setdefaultencoding('utf-8')

import tensorflow as tf
import numpy as np

CROSS_VALID_INDEX = 3
FEATURE_SIZE = 10 * 3
NUM_CLASS = 6
BATCH_SIZE = 32

# 1. load data
train_data = []
test_data = []
for csv_path in glob.glob("../data/*.csv"):
    with open(csv_path) as f:
        raw_data = f.readlines()
    if CROSS_VALID_INDEX == 0:
        train_data += raw_data[60:]
        test_data += raw_data[:60]
    elif CROSS_VALID_INDEX == 1:
        train_data += raw_data[:60]
        train_data += raw_data[120:]
        test_data += raw_data[60:120]
    elif CROSS_VALID_INDEX == 2:
        train_data += raw_data[:120]
        train_data += raw_data[180:]
        test_data += raw_data[120:180]
    elif CROSS_VALID_INDEX == 3:
        train_data += raw_data[:180]
        test_data += raw_data[180:]

for idx in range(len(train_data)):
    train_data[idx] = np.array(train_data[idx].strip().split(",")).astype(np.float32)
for idx in range(len(test_data)):
    test_data[idx] = np.array(test_data[idx].strip().split(",")).astype(np.float32)

# 2. define model
input_feature = tf.placeholder(
        dtype=tf.float32, 
        shape=[None, FEATURE_SIZE], 
        name='input_feature')

target_class = tf.placeholder(
        dtype=tf.int32, 
        shape=[None],
        name='target_class')

w = tf.Variable(tf.random_normal([FEATURE_SIZE, NUM_CLASS]), name='weights')
b = tf.Variable(tf.zeros([NUM_CLASS], name='bias'))
y = tf.matmul(input_feature, w) + b
#cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y, target_class)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_class, logits=y)
loss = tf.reduce_mean(cross_entropy)
train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

#tf.scalar_summary(self.SUMMARIES_TAG+"/loss", self.loss) 
predict = tf.arg_max(y, 1)  
#accuracy = 100.0 * tf.reduce_sum(tf.to_float(tf.equal(self.predict, self.low_level_category))) / self.BATCH_SIZE
#tf.scalar_summary(self.SUMMARIES_TAG+"/accuracy", self.accuracy)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for idx in range(100000):
        feature_data = np.zeros([BATCH_SIZE, FEATURE_SIZE])
        target_data = np.zeros([BATCH_SIZE])
        for sub_idx in range(BATCH_SIZE):
            feature_data[sub_idx] = train_data[(idx*BATCH_SIZE+sub_idx)%len(train_data)][:30] 
            target_data[sub_idx] = int(train_data[(idx*BATCH_SIZE+sub_idx)%len(train_data)][30]) 

        _, loss_value = sess.run([train_op, loss], feed_dict={
                input_feature:feature_data,
                target_class:target_data
            })
        if idx % 1000 == 0:
            print "%d : %0.2f" % (idx, loss_value)


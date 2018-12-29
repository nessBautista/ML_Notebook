from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

#Create 2 tensors and an addition operation
t1 = tf.constant([1.2,2.3,3.4,4.5])
t2 = tf.random_normal([4])
t3 = t1 + t2
graph1 = tf.get_default_graph()

#Create a second graph and make it the default graph
graph2 = tf.Graph()

#Create 2 tensors in the second graph and a substraction operation
with graph2.as_default():
    t4 = tf.constant([5.6,6.7, 7.8,8.9])
    t5 = tf.random_normal([4])
    t6 = t4 -t5

#Create a session and execute the addition operation
with tf.Session(graph=graph1) as sess:
    print('Addition', sess.run(t3))

#Create a second session and execute the substraction operation
with tf.Session(graph=graph2) as sess:
    print('Substraction', sess.run(t6))

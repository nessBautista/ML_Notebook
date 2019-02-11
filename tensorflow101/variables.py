from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

#Create 2 variables and an operation that adds the variables
v1 = tf.Variable(2)
v2 = tf.Variable(3)
v3 = v1 + v2

#Obtain an operation that will initialize the two variables
init = tf.global_variables_initializer()

#Create a session to execute the initialization and the Addition operation
with tf.Session() as sess:
    sess.run(init)

#Execute the addition operation in a session
    result = sess.run(v3)

#print the result of the operation
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info('Result:{0}'.format(result))

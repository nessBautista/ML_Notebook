from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

msg = tf.string_join(['Hello', 'Tensorflow'])

with tf.Session() as sess:
    print(sess.run(msg))

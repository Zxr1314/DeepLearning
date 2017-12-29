'''
tensorboard helper
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev', stddev)
        #tf.summary.scalar('max', tf.reduce_max(var))
        #tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
    return

def loss_summaries(loss):
    tf.summary.scalar('loss', loss)
    return

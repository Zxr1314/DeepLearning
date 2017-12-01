'''Net Base Class
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import tensorflow as tf
import numpy as np

class Net(object):
    def __init__(self, common_params, net_params):
        '''

        :param common_params:
        :param net_params:
        '''
        self.pretrained_collection = []
        self.trainable_collection = []
        return

    def _variable_on_cpu(self, name, shape, initializer, pretrain=True, train=True):
        '''Helper to create a Variable stored on CPU memory.

        :param name: Name of the Variable, str
        :param shape: Shape of the Variable, list of ints
        :param initializer: Initializer of Variable
        :param pretrain:
        :param train:
        :return: Variable Tensor
        '''
        with tf.device('/cpu:0'):
            var = tf.get_variable(name, shape, dtype=tf.float32, initializer=initializer)
            if pretrain:
                self.pretrained_collection.append(var)
            if train:
                self.trainable_collection.append(var)
        return var

    def _variable_with_weight_decay(self, name, shape, stddev, wd, pretrain=True, train=True):
        '''Helper to create an initialized Variable with weight decay (?)

        :param name:
        :param shape:
        :param stddev:
        :param wd:
        :param pretrain:
        :param train:
        :return: Variable Tensor
        '''
        var = self._variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32),
                                    pretrain=pretrain, train=train)
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def conv2d(self, scope, input, kernel_size, stride=[1,1,1,1], padding='SAME', pretrain=True, train=True):
        '''2-D convulution layer

        :param scope: scope name
        :param input: 4-D tensor [batch_size, height, width, channel]
        :param kernel_size: [height, width, in_channel, out_channel]
        :param stride:
        :param pretrain:
        :param train:
        :return: 4-D tensor
        '''
        with tf.name_scope(scope):
            kernel = self._variable_with_weight_decay(scope+'_weights', shape=kernel_size,
                                                      stddev=1.0, wd=1.0,
                                                      pretrain=pretrain, train=train)
            conv = tf.nn.conv2d(input, kernel, strides=stride, padding=padding)
            bn = tf.contrib.layers.batch_norm(conv, decay=0.999, epsilon=1e-3, is_training=True)
            biases = self._variable_on_cpu(scope+'_biases', kernel_size[3:], tf.constant_initializer(0.0), pretrain, train)
            bias = tf.nn.bias_add(bn, biases)
        return bias

    def conv2d_transpose(self, scope, input, target, kernel_size, stride=[1,1,1,1], pretrain=True, train=True):
        '''

        :param scope:
        :param input:
        :param targer:
        :param kernel_size:
        :param stride:
        :param pretrain:
        :param train:
        :return:
        '''
        with tf.name_scope(scope):
            kernel = self._variable_on_cpu(scope+'_weights', kernel_size, tf.truncated_normal_initializer(stddev=1.0, dtype=tf.float32),
                                           pretrain=pretrain, train=train)
            conv = tf.nn.conv2d_transpose(input, kernel, target, stride, name='deconv')
            biases = self._variable_on_cpu(scope+'_biases', kernel_size[2], tf.constant_initializer(0.0), pretrain, train)
            bias = tf.nn.bias_add(conv, biases)
        return bias

    def conv3d(self, scope, input, kernel_size, stride=[1,1,1,1,1], padding='SAME', pretrain=True, train=True):
        '''3-D convulution layer

        :param scope: scope name
        :param input: 4-D tensor [batch_size, height, width, channel]
        :param kernel_size: [height, width, in_channel, out_channel]
        :param stride:
        :param pretrain:
        :param train:
        :return: 4-D tensor
        '''
        with tf.name_scope(scope):
            kernel = self._variable_with_weight_decay('weights', shape=kernel_size,
                                                      stddev=5e-2, wd=1.0,
                                                      pretrain=pretrain, train=train)
            conv = tf.nn.conv3d(input, kernel, strides=stride, padding=padding)
            bn = tf.contrib.layers.batch_norm(conv, decay=0.999, epsilon=1e-3, is_training=True)
            biases = self._variable_on_cpu('biases', kernel_size[4], tf.constant_initializer(0.0), pretrain, train)
            bias = tf.nn.bias_add(bn, biases)
        return bias

    def conv3d_transpose(self, scope, input, target, kernel_size, stride=[1,1,1,1,1], pretrain=True, train=True):
        '''

        :param scope:
        :param input:
        :param targer:
        :param kernel_size:
        :param stride:
        :param pretrain:
        :param train:
        :return:
        '''
        with tf.name_scope(scope):
            kernel = self._variable_on_cpu('weights', kernel_size, tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32),
                                           pretrain=pretrain, train=train)
            conv = tf.nn.conv3d_transpose(input, kernel, target, stride, name='deconv')
            biases = self._variable_on_cpu('biases', kernel_size[3], tf.constant_initializer(0.0), pretrain, train)
            bias = tf.nn.bias_add(conv, biases)
        return bias

    def leaky_relu(self, input, alpha, name=None):
        '''

        :param input:
        :param alpha:
        :param name:
        :return:
        '''
        with tf.name_scope(name):
            mask = tf.cast((input>0), dtype=tf.float32)
            out = mask*input + alpha*(1-mask)*input
        return out

    def prelu(self, input, pretrain=True, train=True, name=None):
        '''

        :param input:
        :param pretrain:
        :param train:
        :param name:
        :return:
        '''
        with tf.name_scope(name):
            p = self._variable_on_cpu('p', 1, tf.constant_initializer(0.1), pretrain, train)
            mask = tf.cast((input>0), dtype=tf.float32)
            out = mask*input + p*(1-mask)*input
        return out

    def inference(self, images):
        '''
        Definition of the network
        :param images: 4-D or 5-D tensor depending on image dimension
        :return: predicts
        '''
        raise NotImplementedError

    def loss(self, predicts, labels, eval_names):
        '''
        Function to calculate losses and other evaluations
        :param predicts:
        :param labels:
        :param eval_names:
        :return: loss:
        :return: evals:
        '''
        raise NotImplementedError
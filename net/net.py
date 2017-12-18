'''Net Base Class
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import math

import tensorflow as tf
from tensorflow.python.layers.normalization import *
import numpy as np

from utils.tensorboard import *

class Net(object):
    def __init__(self, common_params, net_params):
        '''

        :param common_params:
        :param net_params:
        '''
        self.pretrained_collection = []
        self.trainable_collection = []
        self.all_collection = []
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
            var = tf.get_variable(name, shape, dtype=tf.float32, initializer=initializer, trainable=train)
            variable_summaries(var)
            if pretrain:
                self.pretrained_collection.append(var)
            if train:
                self.trainable_collection.append(var)
            self.all_collection.append(var)
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
        var = self._variable_on_cpu(name, shape, tf.random_normal_initializer(stddev=stddev, dtype=tf.float32),
                                    pretrain=pretrain, train=train)
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def conv2d(self, scope, input, kernel_size, stride=[1,1,1,1], padding='SAME', pretrain=True, train=True, use_bias=True):
        '''

        :param scope:
        :param input:
        :param kernel_size:
        :param stride:
        :param padding:
        :param pretrain:
        :param train:
        :param use_bias:
        :return:
        '''
        with tf.name_scope(scope):
            s = float(kernel_size[0]*kernel_size[1])
            kernel = self._variable_with_weight_decay(scope+'_weights', shape=kernel_size,
                                                      stddev=1.0/math.sqrt(s), wd=1.0,
                                                      pretrain=pretrain, train=train)
            conv = tf.nn.conv2d(input, kernel, strides=stride, padding=padding)
            if use_bias:
                biases = self._variable_on_cpu(scope+'_biases', kernel_size[3:], tf.constant_initializer(0.0), pretrain, train)
                conv = tf.nn.bias_add(conv, biases)
            #bn = tf.contrib.layers.batch_norm(conv, decay=0.999, epsilon=1e-3, is_training=True, trainable=train)
            #bn = tf.layers.batch_normalization(conv, training=True, trainable=train)
            bn = BatchNormalization(momentum=0.999, trainable=train, name=scope+'_bn')
            bn2 = bn.apply(conv, training=True)
            if pretrain:
                self.pretrained_collection.append(bn.beta)
                self.pretrained_collection.append(bn.moving_mean)
                self.pretrained_collection.append(bn.moving_variance)
            if train:
                self.trainable_collection.append(bn.beta)
                self.trainable_collection.append(bn.moving_mean)
                self.trainable_collection.append(bn.moving_variance)
            self.all_collection.append(bn.beta)
            self.all_collection.append(bn.moving_mean)
            self.all_collection.append(bn.moving_variance)
        return bn2

    def conv2d_transpose(self, scope, input, target, kernel_size, stride=[1,1,1,1], pretrain=True, train=True, use_bias=True):
        # type: (str, tf.Tensor, tf.Tensor, list, list, bool, bool, bool) -> tf.Tensor
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
            s = float(kernel_size[0] * kernel_size[1])
            kernel = self._variable_on_cpu(scope+'_weights', kernel_size, tf.random_normal_initializer(stddev=1.0/math.sqrt(s), dtype=tf.float32),
                                           pretrain=pretrain, train=train)
            conv = tf.nn.conv2d_transpose(input, kernel, target, stride, name='deconv')
            if use_bias:
                biases = self._variable_on_cpu(scope+'_biases', kernel_size[2], tf.constant_initializer(0.0), pretrain, train)
                conv = tf.nn.bias_add(conv, biases)
        return conv

    def conv3d(self, scope, input, kernel_size, stride=[1,1,1,1,1], padding='SAME', pretrain=True, train=True, use_bias=True):
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
            s = float(kernel_size[0] * kernel_size[1] * kernel_size[2])
            kernel = self._variable_with_weight_decay(scope+'_weights', shape=kernel_size,
                                                      stddev=1.0/math.sqrt(s), wd=1.0,
                                                      pretrain=pretrain, train=train)
            conv = tf.nn.conv3d(input, kernel, strides=stride, padding=padding, name=scope+'_conv')
            #bn = tf.contrib.layers.batch_norm(conv, decay=0.999, epsilon=1e-3, is_training=True)
            if use_bias:
                biases = self._variable_on_cpu(scope+'_biases', kernel_size[4], tf.constant_initializer(0.0), pretrain, train)
                conv = tf.nn.bias_add(conv, biases)
            bn = BatchNormalization(momentum=0.999, trainable=train, name=scope + '_bn')
            bn2 = bn.apply(conv, training=True)
            if pretrain:
                self.pretrained_collection.append(bn.beta)
                self.pretrained_collection.append(bn.moving_mean)
                self.pretrained_collection.append(bn.moving_variance)
            if train:
                self.trainable_collection.append(bn.beta)
                self.trainable_collection.append(bn.moving_mean)
                self.trainable_collection.append(bn.moving_variance)
            self.all_collection.append(bn.beta)
            self.all_collection.append(bn.moving_mean)
            self.all_collection.append(bn.moving_variance)
        return bn2

    def conv3d_transpose(self, scope, input, target, kernel_size, stride=[1,1,1,1,1], pretrain=True, train=True, use_bias=True):
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
        s = float(kernel_size[0] * kernel_size[1] * kernel_size[2])
        with tf.name_scope(scope):
            kernel = self._variable_on_cpu(scope+'_weights', kernel_size, tf.truncated_normal_initializer(stddev=1.0/math.sqrt(s), dtype=tf.float32),
                                           pretrain=pretrain, train=train)
            conv = tf.nn.conv3d_transpose(input, kernel, target, stride, name=scope+'_deconv')
            if use_bias:
                biases = self._variable_on_cpu(scope+'_biases', kernel_size[3], tf.constant_initializer(0.0), pretrain, train)
                conv = tf.nn.bias_add(conv, biases)
        return conv

    def leaky_relu(self, input, alpha=0.2, name=None):
        '''

        :param input:
        :param alpha:
        :param name:
        :return:
        '''
        with tf.name_scope(name):
            out = tf.where(input > 0.0, input, alpha * input)
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

    def selu(self, input, name=None):
        '''

        :param input:
        :param pretrain:
        :param train:
        :param name:
        :return:
        '''
        with tf.name_scope(name):
            alpha = 1.6732632423543772848170429916717
            scale = 1.0507009873554804934193349852946
            out = scale * tf.where(input > 0.0, input, alpha * tf.nn.elu(input))
        return out

    def swish(self, input, name):
        '''

        :param input:
        :param name:
        :return:
        '''
        with tf.name_scope(name):
            out = input*tf.nn.sigmoid(input)
        return out

    def inference(self, images, **kwargs):
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
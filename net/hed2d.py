'''
Holistically-Nested Edge Detection 2-D
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from net.net import Net

class HED2D(Net):
    def __init__(self, common_params, net_params, name=None):
        super(HED2D, self).__init__(common_params, net_params, name)
        return

    def _conv_block(self, input, target_channel, length, pretrain=True, training=True, name=None):
        output = {}
        input_channel = int(input.get_shape()[3])
        conv0 = self.conv2d(name+'_conv0', input, [3,3,input_channel,target_channel],pretrain=pretrain,train=training, use_bias=False)
        relu = tf.nn.relu(conv0, name=name+'_relu0')
        for i in xrange(1, length):
            conv = self.conv2d(name+'_conv'+str(i), relu, [3,3,target_channel,target_channel], pretrain=pretrain, train=training, use_bias=False)
            relu = tf.nn.relu(conv, name=name+'_relu'+str(i))
        output['out'] = relu
        return output

    def inference(self, images, **kwargs):
        output = {}
        input_shape = images.get_shape()
        if 'training' in kwargs:
            training = kwargs['training']
        else:
            training = True
        if 'pretrain' in kwargs:
            pretrain = kwargs['pretrain']
        else:
            pretrain = False

        conv_block0 = self._conv_block(images, 64, 3, pretrain, training, name=self.name+'conv_block0')
        output['conv_block0'] = conv_block0
        self.conv_block0 = conv_block0['out']
        self.max_pool0 = tf.nn.max_pool(self.conv_block0, [1,2,2,1], [1,2,2,1], padding='VALID', name=self.name+'max_pool0')
        output['max_pool0'] = self.max_pool0

        conv_block1 = self._conv_block(self.max_pool0, 128, 2, pretrain, training, name=self.name+'conv_block1')
        output['conv_block1'] = conv_block1
        self.conv_block1 = conv_block1['out']
        self.max_pool1 = tf.nn.max_pool(self.conv_block1, [1,2,2,1], [1,2,2,1], padding='VALID', name=self.name+'max_pool1')
        output['max_pool1'] = self.max_pool1

        conv_block2 = self._conv_block(self.max_pool1, 128, 3, pretrain, training, name=self.name + 'conv_block2')
        output['conv_block2'] = conv_block2
        self.conv_block2 = conv_block2['out']
        self.max_pool2 = tf.nn.max_pool(self.conv_block2, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID',
                                        name=self.name + 'max_pool2')
        output['max_pool2'] = self.max_pool2

        conv_block3 = self._conv_block(self.max_pool2, 256, 2, pretrain, training, name=self.name + 'conv_block3')
        output['conv_block3'] = conv_block3
        self.conv_block3 = conv_block3['out']
        self.max_pool3 = tf.nn.max_pool(self.conv_block3, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID',
                                        name=self.name + 'max_pool3')
        output['max_pool3'] = self.max_pool3

        conv_block4 = self._conv_block(self.max_pool3, 256, 3, pretrain, training, name=self.name + 'conv_block4')
        output['conv_block4'] = conv_block4
        self.conv_block4 = conv_block4['out']
        self.max_pool4 = tf.nn.max_pool(self.conv_block4, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID',
                                        name=self.name + 'max_pool4')
        output['max_pool4'] = self.max_pool4

        self.outconv0 = self.conv2d(self.name + 'outconv0', self.conv_block0, [1, 1, 64, 1], pretrain=pretrain,
                                    train=training)
        output['outconv0'] = self.outconv0
        self.outconv1 = self.conv2d(self.name + 'outconv1', self.conv_block1, [1, 1, 128, 1], pretrain=pretrain,
                                    train=training)
        output['outconv1'] = self.outconv1
        self.outconv2 = self.conv2d(self.name + 'outconv2', self.conv_block2, [1, 1, 128, 1], pretrain=pretrain,
                                    train=training)
        output['outconv2'] = self.outconv2
        self.outconv3 = self.conv2d(self.name + 'outconv3', self.conv_block3, [1, 1, 256, 1], pretrain=pretrain,
                                    train=training)
        output['outconv3'] = self.outconv3
        self.outconv4 = self.conv2d(self.name + 'outconv4', self.conv_block4, [1, 1, 256, 1], pretrain=pretrain,
                                    train=training)
        output['outconv4'] = self.outconv4

        self.out_resize1 = tf.image.resize_images(self.outconv1, [int(input_shape[1]), int(input_shape[2])])
        self.out_resize2 = tf.image.resize_images(self.outconv2, [int(input_shape[1]), int(input_shape[2])])
        self.out_resize3 = tf.image.resize_images(self.outconv3, [int(input_shape[1]), int(input_shape[2])])
        self.out_resize4 = tf.image.resize_images(self.outconv4, [int(input_shape[1]), int(input_shape[2])])

        self.concat = tf.concat([self.outconv0, self.out_resize1, self.out_resize2, self.out_resize3, self.out_resize4], axis=3, name=self.name+'concat')
        output['concat'] = self.concat
        self.last_conv = self.conv2d(self.name+'last_conv', self.concat, [1,1,5,1], pretrain=pretrain, train=training, use_bias=True)
        output['last_conv'] = self.last_conv
        self.out = tf.nn.sigmoid(self.last_conv, name=self.name+'sigm')
        output['out'] = self.out

        return output

    def loss(self, predicts, labels, eval_names):
        weight0 = labels*self.wtrue+self.wfalse
        label1 = tf.nn.max_pool(labels, [1,2,2,1], [1,2,2,1], padding='VALID', name=self.name+'label_maxpool1')
        weight1 = label1*self.wtrue+self.wfalse
        label2 = tf.nn.max_pool(label1, [1,2,2,1], [1,2,2,1], padding='VALID', name=self.name+'label_maxpool2')
        weight2 = label2*self.wtrue+self.wfalse
        label3 = tf.nn.max_pool(label2, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID', name=self.name + 'label_maxpool3')
        weight3 = label3 * self.wtrue + self.wfalse
        label4 = tf.nn.max_pool(label3, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID', name=self.name + 'label_maxpool4')
        weight4 = label4 * self.wtrue + self.wfalse
        loss_out = tf.losses.sigmoid_cross_entropy(labels, self.out, weights=weight0)
        loss0 = tf.losses.sigmoid_cross_entropy(labels, self.outconv0, weights=weight0)
        loss1 = tf.losses.sigmoid_cross_entropy(label1, self.outconv1, weights=weight1)
        loss2 = tf.losses.sigmoid_cross_entropy(label2, self.outconv2, weights=weight2)
        loss3 = tf.losses.sigmoid_cross_entropy(label3, self.outconv3, weights=weight3)
        loss4 = tf.losses.sigmoid_cross_entropy(label4, self.outconv4, weights=weight4)
        loss = loss_out+loss0+loss1+loss2+loss3+loss4

        evals = {}
        if eval_names is not None:
            seg = tf.round(predicts)
            if 'accuracy' in eval_names:
                evals['accuracy'] = tf.reduce_mean(tf.cast(tf.equal(seg, labels), tf.float32))
            TP = tf.cast(tf.count_nonzero(seg * labels), dtype=tf.float32)
            FP = tf.cast(tf.count_nonzero((1 - seg) * labels), dtype=tf.float32)
            FN = tf.cast(tf.count_nonzero(seg * (1 - labels)), dtype=tf.float32)
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            f1 = 2 * precision * recall / (precision + recall)
            if 'precision' in eval_names:
                evals['precision'] = precision
            if 'recall' in eval_names:
                evals['recall'] = recall
            if 'f1' in eval_names:
                evals['f1'] = f1
            if 'dice' in eval_names:
                evals['dice'] = 2 * TP / (2 * TP + FP + FN)
        return loss, evals
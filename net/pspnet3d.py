'''
PSP net 2-D
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import numpy as np

from net.net import Net
from net.resnet3d import ResNet3D

class PSPnet3D(Net):
    def __init__(self, common_params, net_params, name=None):
        super(PSPnet3D, self).__init__(common_params, net_params)
        self.width = common_params['width']
        self.height = common_params['height']
        self.batch_size = common_params['batch_size']
        if net_params.has_key('weight_true'):
            self.wtrue = net_params['weight_true']
        else:
            self.wtrue = 0
        if net_params.has_key('weight_false'):
            self.wfalse = net_params['weight_false']
        else:
            self.wfalse = 1
        if name is None:
            self.name = ''
        else:
            self.name = name + '/'
        self.resnet = ResNet3D(common_params, net_params, self.name+'res')
        return

    def interp_block(self, input, level, feature_map_shape, str_lvl=1, pretrain=False, training=True, name=None):
        str_lvl = str(str_lvl)
        input_channel = int(input.get_shape()[4])
        input_shape = tf.shape(input)
        output = {}
        with tf.name_scope(name):
            prev_layer = tf.nn.avg_pool3d(input, [1, 2**level, 2**level, 2**level, 1], [1, 2**level, 2**level, 2**level, 1], padding='VALID', name=name+'_avg_pool')
            output['avg_pool'] = prev_layer
            prev_layer = self.conv3d(name+'_conv', prev_layer, [1,1,1,input_channel,256], pretrain=pretrain, train=training, use_bias=False)
            output['conv'] = prev_layer
            prev_layer = tf.nn.relu(prev_layer, name=name+'_relu')
            output['relu'] = prev_layer
            layer_shape = tf.shape(prev_layer)
            #prev_layer = tf.image.resize_images(prev_layer, feature_map_shape)
            prev_layer = self.conv3d_transpose(name + '_resize1', prev_layer,
                                               [input_shape[0], input_shape[1], layer_shape[2], layer_shape[3], 256],
                                               [2 ** level, 1, 1, 256, 256], [1, 2 ** level, 1, 1, 1],
                                               pretrain=pretrain, train=training, use_bias=False)
            prev_layer = self.conv3d_transpose(name + '_resize2', prev_layer,
                                               [input_shape[0], input_shape[1], input_shape[2], layer_shape[3], 256],
                                               [1, 2**level, 1, 256, 256], [1, 1, 2**level, 1, 1], pretrain=pretrain,
                                               train=training, use_bias=False)
            prev_layer = self.conv3d_transpose(name + '_resize3', prev_layer,
                                               [input_shape[0], input_shape[1], input_shape[2], input_shape[3], 256],
                                               [1, 1, 2 ** level, 256, 256], [1, 1, 1, 2 ** level, 1],
                                               pretrain=pretrain, train=training, use_bias=False)
            output['out'] = prev_layer
        return output

    def build_pyramid_pooling_module(self, input, input_shape, pretrain=False, training=True):
        #feature_map_size = tuple(int(math.ceil(input_dim/8.0)) for input_dim in input_shape)
        feature_map_size = input.get_shape()[1:4]
        output = {}
        interp_block1 = self.interp_block(input, 4, feature_map_size, str_lvl=1, pretrain=pretrain, training=training, name=self.name+'interp_block1')
        output['interp_block1'] = interp_block1
        interp_block2 = self.interp_block(input, 3, feature_map_size, str_lvl=2, pretrain=pretrain, training=training, name=self.name+'interp_block2')
        output['interp_block2'] = interp_block2
        interp_block3 = self.interp_block(input, 2, feature_map_size, str_lvl=3, pretrain=pretrain, training=training, name=self.name+'interp_block3')
        output['interp_block3'] = interp_block3
        interp_block4 = self.interp_block(input, 1, feature_map_size, str_lvl=4, pretrain=pretrain, training=training, name=self.name+'interp_block4')
        output['interp_block4'] = interp_block4

        #res = tf.concat([input, interp_block1, interp_block2, interp_block3, interp_block6], axis=3, name='concat')
        try:
            res = tf.concat([input, interp_block1['out'], interp_block2['out'], interp_block3['out'], interp_block4['out']], axis=4, name=self.name+'concat')
            output['out'] = res
        except:
            res = tf.concat_v2([input, interp_block1['out'], interp_block2['out'], interp_block3['out'], interp_block4['out']], axis=4, name=self.name+'concat')
            output['out'] = res
        return output

    def inference(self, images, **kwargs):
        shape = images.get_shape()
        shape2 = tf.shape(images)
        output = {}
        if 'training' in  kwargs:
            training = kwargs['training']
        else:
            training = True
        if 'pretrain' in kwargs:
            pretrain = kwargs['pretrain']
        else:
            pretrain = True
        res = self.resnet.inference(images, pretrain=pretrain, training=training)
        output['resnet'] = res
        input_shape = res['relu4'].get_shape()[1:4]
        psp = self.build_pyramid_pooling_module(res['relu4'], input_shape, pretrain=pretrain, training=training)
        output['psp'] = psp
        input_channel = int(psp['out'].get_shape()[4])
        x = self.conv3d(self.name+'conv5_4', psp['out'], [3,3,3,input_channel,384], use_bias=False, pretrain=pretrain, train=training)
        output['conv5_4'] = x
        x = tf.nn.relu(x, name=self.name+'relu5_4')
        output['relu1'] = x
        if 'keep_prob' not in kwargs:
            raise Exception("Keep_prob for dropout layer not given!")
        x = tf.nn.dropout(x, kwargs['keep_prob'])
        output['dropout'] = x
        out_shape = [shape2[0], int(int(shape[1])/4), int(int(shape[2])/4), int(int(shape[3])/4), 192]
        x = self.conv3d_transpose(self.name+'deconv1', x, out_shape, [2, 2, 2, 192, 384], stride=[1,2,2,2,1], pretrain=pretrain, train=training)
        output['deconv1'] = x
        out_shape = [shape2[0], int(int(shape[1]) / 2), int(int(shape[2]) / 2), int(int(shape[3]) / 2), 96]
        x = self.conv3d_transpose(self.name+'deconv2', x, out_shape, [2, 2, 2, 96, 192], stride=[1, 2, 2, 2, 1], pretrain=pretrain, train=training)
        output['deconv2'] = x
        out_shape = [shape2[0], int(shape[1]), int(shape[2]), int(shape[3]), 48]
        x = self.conv3d_transpose(self.name+'deconv3', x, out_shape, [2, 2, 2, 48, 96], stride=[1, 2, 2, 2, 1], pretrain=pretrain, train=training)
        output['deconv3'] = x
        x = self.conv3d(self.name+'conv6', x, [1,1,1,48,1], pretrain=pretrain, train=training)
        output['conv6'] = x
        self.last_conv = x

        #x = self.conv2d_transpose('conv6', x, shape, [8,8,1,512])
        sigm = tf.nn.sigmoid(x, name=self.name+'sigm')
        output['out'] = sigm
        self.pretrained_collection += self.resnet.pretrained_collection
        self.trainable_collection += self.resnet.trainable_collection
        self.all_collection += self.resnet.all_collection
        return output

    def loss(self, predicts, labels, eval_names):
        weight = labels*self.wtrue+self.wfalse
        loss = tf.losses.absolute_difference(labels, predicts, weights=weight)
        #loss = -self.wtrue*self.last_conv*labels+self.wfalse*tf.log(tf.exp(self.last_conv)+1.0)+(self.wtrue-self.wfalse)*labels*tf.log(tf.exp(self.last_conv)+1)
        loss = tf.reduce_mean(loss)
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
        return loss, evals
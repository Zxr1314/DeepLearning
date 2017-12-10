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
from net.resnet2d import ResNet2D


class PSPnet2D(Net):
    def __init__(self, common_params, net_params):
        super(PSPnet2D, self).__init__(common_params, net_params)
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
        self.resnet = ResNet2D(common_params, net_params)
        return

    def interp_block(self, input, level, feature_map_shape, str_lvl=1, name=None):
        str_lvl = str(str_lvl)
        input_channel = int(input.get_shape()[3])
        output = {}
        with tf.name_scope(name):
            prev_layer = tf.nn.avg_pool(input, [1, 10 * level, 10 * level, 1], [1, 10 * level, 10 * level, 1],
                                        padding='VALID', name=name + '_avg_pool')
            output['avg_pool'] = prev_layer
            prev_layer = self.conv2d(name + '_conv', prev_layer, [1, 1, input_channel, 512])
            output['conv'] = prev_layer
            prev_layer = tf.nn.relu(prev_layer, name=name + '_relu')
            output['relu'] = prev_layer
            prev_layer = tf.image.resize_images(prev_layer, feature_map_shape)
            output['out'] = prev_layer
        return output

    def build_pyramid_pooling_module(self, input, input_shape):
        #feature_map_size = tuple(int(math.ceil(input_dim/8.0)) for input_dim in input_shape)
        feature_map_size = input.get_shape()[1:3]
        output = {}
        interp_block1 = self.interp_block(input, 6, feature_map_size, str_lvl=1, name='interp_block1')
        output['interp_block1'] = interp_block1
        interp_block2 = self.interp_block(input, 3, feature_map_size, str_lvl=2, name='interp_block2')
        output['interp_block2'] = interp_block2
        interp_block3 = self.interp_block(input, 2, feature_map_size, str_lvl=3, name='interp_block3')
        output['interp_block3'] = interp_block3
        interp_block6 = self.interp_block(input, 1, feature_map_size, str_lvl=6, name='interp_block6')
        output['interp_block6'] = interp_block6

        #res = tf.concat([input, interp_block1, interp_block2, interp_block3, interp_block6], axis=3, name='concat')
        try:
            res = tf.concat([input, interp_block1['out'], interp_block2['out'], interp_block3['out'], interp_block6['out']], axis=3, name='concat')
            output['out'] = res
        except:
            res = tf.concat_v2([input, interp_block1['out'], interp_block2['out'], interp_block3['out'], interp_block6['out']], axis=3, name='concat')
            output['out'] = res
        return output

    def inference(self, images, **kwargs):
        output = {}
        res = self.resnet.inference(images)
        output['resnet'] = res
        input_shape = res['relu4'].get_shape()[1:3]
        psp = self.build_pyramid_pooling_module(res['relu4'], input_shape)
        output['psp'] = psp
        input_channel = int(psp['out'].get_shape()[3])
        x = self.conv2d('conv5_4', psp['out'], [3, 3, input_channel, 512], use_bias=False)
        output['conv5_4'] = x
        x = tf.nn.relu(x)
        output['relu1'] = x
        if 'keep_prob' not in kwargs:
            raise Exception("Keep_prob for dropout layer not given!")
        x = tf.nn.dropout(x, kwargs['keep_prob'])
        output['dropout'] = x
        x = tf.image.resize_images(x, [self.height, self.width])
        output['resize'] = x
        x = self.conv2d('conv6', x, [1, 1, 512, 1])
        output['conv6'] = x

        # x = self.conv2d_transpose('conv6', x, shape, [8,8,1,512])
        sigm = tf.nn.sigmoid(x, name='sigm')
        output['out'] = sigm
        return output

    def loss(self, predicts, labels, eval_names):
        weight = labels * self.wtrue + self.wfalse
        loss = tf.losses.absolute_difference(labels, predicts, weights=weight)
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

class PSPnet2D2(Net):
    def __init__(self, common_params, net_params):
        super(PSPnet2D2, self).__init__(common_params, net_params)
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
        self.resnet = ResNet2D(common_params, net_params)
        return

    def interp_block(self, input, level, feature_map_shape, str_lvl=1, pretrain=False, training=True, name=None):
        str_lvl = str(str_lvl)
        input_channel = int(input.get_shape()[3])
        output = {}
        with tf.name_scope(name):
            prev_layer = tf.nn.avg_pool(input, [1, 10*level, 10*level, 1], [1, 10*level, 10*level, 1], padding='VALID', name=name+'_avg_pool')
            output['avg_pool'] = prev_layer
            prev_layer = self.conv2d(name+'_conv', prev_layer, [1,1,input_channel,512], pretrain=pretrain, train=training)
            output['conv'] = prev_layer
            prev_layer = tf.nn.relu(prev_layer, name=name+'_relu')
            output['relu'] = prev_layer
            prev_layer = tf.image.resize_images(prev_layer, feature_map_shape)
            output['out'] = prev_layer
        return output

    def build_pyramid_pooling_module(self, input, input_shape, pretrain=False, training=True):
        #feature_map_size = tuple(int(math.ceil(input_dim/8.0)) for input_dim in input_shape)
        feature_map_size = input.get_shape()[1:3]
        output = {}
        interp_block1 = self.interp_block(input, 6, feature_map_size, str_lvl=1, pretrain=pretrain, training=training, name='interp_block1')
        output['interp_block1'] = interp_block1
        interp_block2 = self.interp_block(input, 3, feature_map_size, str_lvl=2, pretrain=pretrain, training=training, name='interp_block2')
        output['interp_block2'] = interp_block2
        interp_block3 = self.interp_block(input, 2, feature_map_size, str_lvl=3, pretrain=pretrain, training=training, name='interp_block3')
        output['interp_block3'] = interp_block3
        interp_block6 = self.interp_block(input, 1, feature_map_size, str_lvl=6, pretrain=pretrain, training=training, name='interp_block6')
        output['interp_block6'] = interp_block6

        #res = tf.concat([input, interp_block1, interp_block2, interp_block3, interp_block6], axis=3, name='concat')
        try:
            res = tf.concat([input, interp_block1['out'], interp_block2['out'], interp_block3['out'], interp_block6['out']], axis=3, name='concat')
            output['out'] = res
        except:
            res = tf.concat_v2([input, interp_block1['out'], interp_block2['out'], interp_block3['out'], interp_block6['out']], axis=3, name='concat')
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
        input_shape = res['relu4'].get_shape()[1:3]
        psp = self.build_pyramid_pooling_module(res['relu4'], input_shape, pretrain=pretrain, training=training)
        output['psp'] = psp
        input_channel = int(psp['out'].get_shape()[3])
        x = self.conv2d('conv5_4', psp['out'], [3,3,input_channel,512], use_bias=False, pretrain=pretrain, train=training)
        output['conv5_4'] = x
        x = tf.nn.relu(x)
        output['relu1'] = x
        if 'keep_prob' not in kwargs:
            raise Exception("Keep_prob for dropout layer not given!")
        x = tf.nn.dropout(x, kwargs['keep_prob'])
        output['dropout'] = x
        out_shape = [shape2[0], int(int(shape[1])/4), int(int(shape[2])/4), 256]
        x = self.conv2d_transpose('deconv1', x, out_shape, [2, 2, 256, 512], stride=[1,2,2,1], pretrain=pretrain, train=training)
        output['deconv1'] = x
        out_shape = [shape2[0], int(int(shape[1]) / 2), int(int(shape[2]) / 2), 128]
        x = self.conv2d_transpose('deconv2', x, out_shape, [2, 2, 128, 256], stride=[1, 2, 2, 1], pretrain=pretrain, train=training)
        output['deconv2'] = x
        out_shape = [shape2[0], int(shape[1]), int(shape[2]), 64]
        x = self.conv2d_transpose('deconv3', x, out_shape, [2, 2, 64, 128], stride=[1, 2, 2, 1], pretrain=pretrain, train=training)
        output['deconv3'] = x
        x = self.conv2d('conv6', x, [1,1,64,1], pretrain=pretrain, train=training)
        output['conv6'] = x

        #x = self.conv2d_transpose('conv6', x, shape, [8,8,1,512])
        sigm = tf.nn.sigmoid(x, name='sigm')
        output['out'] = sigm
        self.pretrained_collection += self.resnet.pretrained_collection
        self.trainable_collection += self.resnet.trainable_collection
        return output

    def loss(self, predicts, labels, eval_names):
        weight = labels*self.wtrue+self.wfalse
        loss = tf.losses.absolute_difference(labels, predicts, weights=weight)
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

class PSPnet2D3(Net):
    def __init__(self, common_params, net_params):
        super(PSPnet2D3, self).__init__(common_params, net_params)
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
        self.pspnet2 = PSPnet2D2(common_params, net_params)
        return

    def inference(self, images, **kwargs):
        if 'keep_prob' not in kwargs:
            raise Exception("Keep_prob for dropout layer not given!")
        if 'former_train' in  kwargs:
            former_train = kwargs['former_train']
        else:
            former_train = False
        if 'training' in  kwargs:
            training = kwargs['training']
        else:
            training = True
        if 'pretrain' in kwargs:
            pretrain = kwargs['pretrain']
        else:
            pretrain = False
        output = {}
        psp = self.pspnet2.inference(images, keep_prob=kwargs['keep_prob'], pretrain=True, training=former_train)
        output['psp'] = psp
        psp_cat = tf.concat([images, psp['conv6']], axis=3, name='psp_concat')
        output['psp_cat'] = psp_cat
        conv = self.conv2d('add_conv1', psp_cat, [3,3,2,16], pretrain=pretrain, train=training)
        output['add_conv1'] = conv
        relu = tf.nn.relu(conv, name='add_relu1')
        output['add_relu1'] = relu
        conv = self.conv2d('add_conv2', relu, [3,3,16,16], pretrain=pretrain, train=training)
        output['add_relu2'] = conv
        relu = tf.nn.relu(conv, name='add_relu2')
        output['add_relu2'] = relu
        conv = self.conv2d('add_conv3', relu, [1,1,16,1], pretrain=pretrain, train=training, use_bias=True)
        output['add_conv3'] = conv
        sigm = tf.nn.sigmoid(conv, name='add_sigm')
        output['out'] = sigm
        self.pretrained_collection += self.pspnet2.pretrained_collection
        self.trainable_collection += self.pspnet2.trainable_collection

        return output

    def loss(self, predicts, labels, eval_names):
        weight = labels*self.wtrue+self.wfalse
        loss = tf.losses.absolute_difference(labels, predicts, weights=weight)
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
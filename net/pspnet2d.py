'''
PSP net 2-D
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

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
        with tf.name_scope(name):
            prev_layer = tf.nn.avg_pool(input, [1, 10*level, 10*level, 1], [1, 10*level, 10*level, 1], padding='VALID', name=name+'_avg_pool')
            prev_layer = self.conv2d(name+'_conv', prev_layer, [1,1,input_channel,512])
            prev_layer = tf.nn.relu(prev_layer, name=name+'_relu')
            prev_layer = tf.image.resize_images(prev_layer, feature_map_shape)
        return prev_layer

    def build_pyramid_pooling_module(self, input, input_shape):
        #feature_map_size = tuple(int(math.ceil(input_dim/8.0)) for input_dim in input_shape)
        feature_map_size = input.get_shape()[1:3]

        interp_block1 = self.interp_block(input, 6, feature_map_size, str_lvl=1, name='interp_block1')
        interp_block2 = self.interp_block(input, 3, feature_map_size, str_lvl=2, name='interp_block2')
        interp_block3 = self.interp_block(input, 2, feature_map_size, str_lvl=3, name='interp_block3')
        interp_block6 = self.interp_block(input, 1, feature_map_size, str_lvl=6, name='interp_block6')

        #res = tf.concat([input, interp_block1, interp_block2, interp_block3, interp_block6], axis=3, name='concat')
        try:
            res = tf.concat([input, interp_block1, interp_block2, interp_block3, interp_block6], axis=3, name='concat')
        except:
            res = tf.concat_v2([input, interp_block1, interp_block2, interp_block3, interp_block6], axis=3, name='concat')
        return res

    def inference(self, images):
        shape = tf.shape(images)
        res = self.resnet.inference(images)
        input_shape = res.get_shape()[1:3]
        psp = self.build_pyramid_pooling_module(res, input_shape)
        input_channel = int(psp.get_shape()[3])
        x = self.conv2d('conv5_4', psp, [3,3,input_channel,512], use_bias=False)
        x = tf.nn.relu(x)
        x = tf.nn.dropout(x, 0.1)
        x = self.conv2d('conv6', x, [1,1,512,1])
        #x = self.conv2d_transpose('conv6', x, shape, [8,8,1,512])
        x = tf.image.resize_images(x, [self.height, self.width])
        sigm = tf.nn.sigmoid(x, name='sigm')
        return sigm

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
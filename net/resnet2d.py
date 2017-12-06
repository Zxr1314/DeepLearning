'''
ResNet 2-D
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from net.net import Net

class ResNet2D(Net):
    def __init__(self, common_params, net_params):
        super(ResNet2D, self).__init__(common_params, net_params)
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
        self.layers = net_params['layers']
        return

    def residual_conv(self, input, level, pad=1, lvl=1, sub_lvl=1, modify_stride=False, name=None):
        lvl = str(lvl)
        sub_lvl = str(sub_lvl)
        input_channel = int(input.get_shape()[3])

        with tf.name_scope(name):
            if modify_stride:
                prev = self.conv2d("reduce_"+lvl+'_'+sub_lvl, input, [1,1,input_channel,64*level], stride=[1,2,2,1])
            else:
                prev = self.conv2d("reduce_"+lvl+'_'+sub_lvl, input, [1,1,input_channel,64*level], stride=[1,1,1,1])
            prev = tf.nn.relu(prev)
            prev = self.conv2d("3x3_"+lvl+"_"+sub_lvl, prev, [3,3,64*level,64*level], stride=[1,1,1,1], use_bias=False)
            prev = tf.nn.relu(prev)
            prev = self.conv2d("1x1_"+lvl+'_'+sub_lvl, prev, [1,1,64*level,256*level], stride=[1,1,1,1], use_bias=False)
        return prev

    def short_convolution_branch(self, input, level, lvl=1, sub_lvl=1, modify_stride=False, name=None):
        lvl = str(lvl)
        sub_lvl = str(sub_lvl)
        input_channel = int(input.get_shape()[3])

        with tf.name_scope(name):
            if modify_stride:
                prev = self.conv2d(name+'_conv', input, [1,1,input_channel,256*level], stride=[1,2,2,1], use_bias=False)
            else:
                prev = self.conv2d(name+'_conv', input, [1,1,input_channel,256*level], stride=[1,1,1,1], use_bias=False)
        return prev

    def empty_branch(self, input):
        return input

    def residual_short(self, input, level, pad=1, lvl=1, sub_lvl=1, modify_stride=False, name=None):
        with tf.name_scope(name):
            prev_layer = tf.nn.relu(input, name=name+'_relu')
            block_1 = self.residual_conv(prev_layer, level, pad=pad, lvl=lvl, sub_lvl=sub_lvl, modify_stride=modify_stride, name=name+'_block_1')
            block_2 = self.short_convolution_branch(prev_layer, level, lvl=lvl, sub_lvl=sub_lvl, modify_stride=modify_stride, name=name+'_block_2')
            added = tf.add(block_1, block_2, name=name+'_add')
        return added

    def residual_empty(self, input, level, pad=1, lvl=1, sub_lvl=1, name=None):
        with tf.name_scope(name):
            prev_layer = tf.nn.relu(input, name=name+'_relu')
            block_1 = self.residual_conv(prev_layer, level, pad, lvl, sub_lvl, name=name+'_block_1')
            block_2 = self.empty_branch(prev_layer)
            added = tf.add(block_1, block_2, name=name+'_add')
        return added

    def inference(self, images, **kwargs):
        input_channel = int(images.get_shape()[3])
        conv1 = self.conv2d('conv1', images, [3,3,input_channel,64], stride=[1,2,2,1], use_bias=False)
        relu1 = tf.nn.relu(conv1, name='relu1')
        conv2 = self.conv2d('conv2', relu1, [3,3,64,64], use_bias=False)
        relu2 = tf.nn.relu(conv2, name='relu2')
        conv3 = self.conv2d('conv3', relu2, [3,3,64,128], use_bias=False)
        relu3 = tf.nn.relu(conv3, name='relu3')
        pool1 = tf.nn.max_pool(relu3, [1,3,3,1], strides=[1,2,2,1], padding='SAME')
        # 2_1-2_3
        res = self.residual_short(pool1, 1, pad=1, lvl=2, sub_lvl=1, name='res_short1')
        for i in range(2):
            res = self.residual_empty(res, 1, pad=1, lvl=2, sub_lvl=i+2, name='res_empty1_'+str(i))
        # 3_1-3_3
        res = self.residual_short(res, 2, pad=2, lvl=3, sub_lvl=1, name='res_short2', modify_stride=True)
        for i in range(3):
            res = self.residual_empty(res, 2, pad=1, lvl=3, sub_lvl=i+2, name='res_empty2_'+str(i))
        if self.layers is 50:
            res = self.residual_short(res, 4, pad=2, lvl=4, sub_lvl=1, name='res_short3')
            for i in xrange(5):
                res = self.residual_empty(res, 4, pad=2, lvl=4, sub_lvl=i+2, name='res_empty3_'+str(i))
        elif self.layers is 101:
            res = self.residual_short(res, 4, pad=2, lvl=4, sub_lvl=1, name='res_short3')
            for i in xrange(22):
                res = self.residual_empty(res, 4, pad=2, lvl=4, sub_lvl=i + 2, name='res_empty3_'+str(i))
        else:
            raise NotImplementedError
        # 5_1-5_3
        res = self.residual_short(res, 8, pad=4, lvl=5, sub_lvl=1, name='res_short4')
        for i in xrange(2):
            res = self.residual_empty(res, 8, pad=4, lvl=5, sub_lvl=i+2, name='res_empty4_'+str(i))
        res = tf.nn.relu(res)
        return res


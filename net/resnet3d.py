'''
ResNet 3-D
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from net.net import Net

class ResNet3D(Net):
    def __init__(self, common_params, net_params, name=None):
        super(ResNet3D, self).__init__(common_params, net_params)
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
        if name is None:
            self.name = None
        else:
            self.name = name + '/'
        return

    def residual_conv(self, input, level, pad=1, lvl=1, sub_lvl=1, modify_stride=False, pretrain=False, training=True, name=None):
        lvl = str(lvl)
        sub_lvl = str(sub_lvl)
        input_channel = int(input.get_shape()[4])
        output = {}
        with tf.name_scope(name):
            if modify_stride:
                prev = self.conv3d(name+"reduce_"+lvl+'_'+sub_lvl, input, [1,1,1,input_channel,64*level], stride=[1,2,2,2,1], pretrain=pretrain, train=training)
                output['conv1'] = prev
            else:
                prev = self.conv3d(name+"reduce_"+lvl+'_'+sub_lvl, input, [1,1,1,input_channel,64*level], stride=[1,1,1,1,1], pretrain=pretrain, train=training)
                output['conv1'] = prev
            prev = tf.nn.relu(prev)
            output['relu1'] = prev
            prev = self.conv3d(name+"3x3_"+lvl+"_"+sub_lvl, prev, [3,3,3,64*level,64*level], stride=[1,1,1,1,1], use_bias=False, pretrain=pretrain, train=training)
            output['conv2'] = prev
            prev = tf.nn.relu(prev)
            output['relu2'] = prev
            prev = self.conv3d(name+"1x1_"+lvl+'_'+sub_lvl, prev, [1,1,1,64*level,256*level], stride=[1,1,1,1,1], use_bias=False, pretrain=pretrain, train=training)
            output['out'] = prev
        return output

    def short_convolution_branch(self, input, level, lvl=1, sub_lvl=1, modify_stride=False, pretrain=False, training=True, name=None):
        lvl = str(lvl)
        sub_lvl = str(sub_lvl)
        input_channel = int(input.get_shape()[4])
        output = {}
        with tf.name_scope(name):
            if modify_stride:
                prev = self.conv3d(name+'_conv', input, [1,1,1,input_channel,256*level], stride=[1,2,2,2,1], use_bias=False, pretrain=pretrain, train=training)
                output['out'] = prev
            else:
                prev = self.conv3d(name+'_conv', input, [1,1,1,input_channel,256*level], stride=[1,1,1,1,1], use_bias=False, pretrain=pretrain, train=training)
                output['out'] = prev
        return output

    def empty_branch(self, input):
        output = {}
        output['out'] = input
        return output

    def residual_short(self, input, level, pad=1, lvl=1, sub_lvl=1, modify_stride=False, pretrain=False, training=True, name=None):
        output = {}
        with tf.name_scope(name):
            prev_layer = tf.nn.relu(input, name=name+'_relu')
            block_1 = self.residual_conv(prev_layer, level, pad=pad, lvl=lvl, sub_lvl=sub_lvl, modify_stride=modify_stride, pretrain=pretrain, training=training, name=name+'_block_1')
            output['block_1'] = block_1
            block_2 = self.short_convolution_branch(prev_layer, level, lvl=lvl, sub_lvl=sub_lvl, modify_stride=modify_stride, pretrain=pretrain, training=training, name=name+'_block_2')
            output['block_2'] = block_2
            added = tf.add(block_1['out'], block_2['out'], name=name+'_add')
            output['out'] = added
        return output

    def residual_empty(self, input, level, pad=1, lvl=1, sub_lvl=1, pretrain=False, training=True, name=None):
        output = {}
        with tf.name_scope(name):
            prev_layer = tf.nn.relu(input, name=name+'_relu')
            output['relu'] = prev_layer
            block_1 = self.residual_conv(prev_layer, level, pad, lvl, sub_lvl, pretrain=pretrain, training=training, name=name+'_block_1')
            output['block_1'] = block_1
            block_2 = self.empty_branch(prev_layer)
            output['block_2'] = block_2
            added = tf.add(block_1['out'], block_2['out'], name=name+'_add')
            output['out'] = added
        return output

    def inference(self, images, **kwargs):
        output = {}
        input_channel = int(images.get_shape()[4])
        if 'training' in  kwargs:
            training = kwargs['training']
        else:
            training = True
        if 'pretrain' in  kwargs:
            pretrain = kwargs['pretrain']
        else:
            pretrain = False
        conv1 = self.conv3d(self.name+'conv1', images, [3,3,3,input_channel,64], stride=[1,2,2,2,1], use_bias=False, pretrain=pretrain, train=training)
        output['conv1'] = conv1
        relu1 = tf.nn.relu(conv1, name=self.name+'relu1')
        output['relu1'] = relu1
        conv2 = self.conv3d(self.name+'conv2', relu1, [3,3,3,64,64], use_bias=False, pretrain=pretrain, train=training)
        output['conv2'] = conv2
        relu2 = tf.nn.relu(conv2, name=self.name+'relu2')
        output['relu2'] = relu2
        conv3 = self.conv3d(self.name+'conv3', relu2, [3,3,3,64,128], use_bias=False, pretrain=pretrain, train=training)
        output['conv3'] = conv3
        relu3 = tf.nn.relu(conv3, name=self.name+'relu3')
        output['relu3'] = relu3
        pool1 = tf.nn.max_pool3d(relu3, [1,3,3,3,1], strides=[1,2,2,2,1], padding='SAME', name=self.name+'maxpool')
        output['pool1'] = pool1
        # 2_1-2_3
        res = self.residual_short(pool1, 1, pad=1, lvl=2, sub_lvl=1, pretrain=pretrain, training=training, name=self.name+'res_short1')
        output['res_short1'] = res
        for i in range(2):
            res = self.residual_empty(res['out'], 1, pad=1, lvl=2, sub_lvl=i+2, pretrain=pretrain, training=training, name=self.name+'res_empty1_'+str(i))
            output['res_empty1_' + str(i)] = res
        # 3_1-3_3
        res = self.residual_short(res['out'], 2, pad=2, lvl=3, sub_lvl=1, pretrain=pretrain, training=training, name=self.name+'res_short2', modify_stride=True)
        output['res_short2'] = res
        for i in range(3):
            res = self.residual_empty(res['out'], 2, pad=1, lvl=3, sub_lvl=i+2, pretrain=pretrain, training=training, name=self.name+'res_empty2_'+str(i))
            output['res_empty2_'+str(i)] = res
        if self.layers is 50:
            res = self.residual_short(res['out'], 4, pad=2, lvl=4, sub_lvl=1, pretrain=pretrain, training=training, name=self.name+'res_short3')
            output['res_short3'] = res
            for i in xrange(5):
                res = self.residual_empty(res['out'], 4, pad=2, lvl=4, sub_lvl=i+2, pretrain=pretrain, training=training, name=self.name+'res_empty3_'+str(i))
                output['res_empty3_' + str(i)] = res
        elif self.layers is 101:
            res = self.residual_short(res['out'], 4, pad=2, lvl=4, sub_lvl=1, pretrain=pretrain, training=training, name=self.name+'res_short3')['out']
            output['res_short3'] = res
            for i in xrange(22):
                res = self.residual_empty(res['out'], 4, pad=2, lvl=4, sub_lvl=i + 2, pretrain=pretrain, training=training, name=self.name+'res_empty3_'+str(i))
                output['res_empty3_' + str(i)] = res
        else:
            raise NotImplementedError
        # 5_1-5_3
        res = self.residual_short(res['out'], 8, pad=4, lvl=5, sub_lvl=1, pretrain=pretrain, training=training, name=self.name+'res_short4')
        output['res_short4'] = res
        for i in xrange(2):
            res = self.residual_empty(res['out'], 8, pad=4, lvl=5, sub_lvl=i+2, pretrain=pretrain, training=training, name=self.name+'res_empty4_'+str(i))
            output['res_empty4_' + str(i)] = res
        res = tf.nn.relu(res['out'])
        output['relu4'] = res
        conv = self.conv3d(self.name+'dense', res, [16,16,16,int(res.get_shape()[4]),1], padding='VALID', pretrain=pretrain, train=training, use_bias=True)
        self.last_conv = conv
        out = tf.nn.sigmoid(conv, name=self.name+'sigm')
        output['out'] = out
        return output


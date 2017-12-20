'''
channel net 2-D
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from net.resnet2d import ResNet2D2
from net.pspnet2d import PSPnet2D2
from net.hed2d import HED2D

class ChanNet2D(PSPnet2D2):
    def __init__(self, common_params, net_params, name=None):
        super(ChanNet2D, self).__init__(common_params, net_params, name)
        self.resnet = ResNet2D2(common_params, net_params, name=self.name+'resnet')
        self.hed = HED2D(common_params, net_params, name=self.name+'hed')
        return

    def build_pyramid_pooling_module(self, input, input_shape, pretrain=False, training=True):
        #feature_map_size = tuple(int(math.ceil(input_dim/8.0)) for input_dim in input_shape)
        feature_map_size = input.get_shape()[1:3]
        output = {}
        interp_block1 = self.interp_block(input, 6, feature_map_size, str_lvl=1, pretrain=pretrain, training=training, name=self.name+'interp_block1')
        output['interp_block1'] = interp_block1
        interp_block2 = self.interp_block(input, 3, feature_map_size, str_lvl=2, pretrain=pretrain, training=training, name=self.name+'interp_block2')
        output['interp_block2'] = interp_block2
        interp_block3 = self.interp_block(input, 2, feature_map_size, str_lvl=3, pretrain=pretrain, training=training, name=self.name+'interp_block3')
        output['interp_block3'] = interp_block3

        #res = tf.concat([input, interp_block1, interp_block2, interp_block3, interp_block6], axis=3, name='concat')
        try:
            res = tf.concat([input, interp_block1['out'], interp_block2['out'], interp_block3['out']], axis=3, name=self.name+'concat')
            output['out'] = res
        except:
            res = tf.concat_v2([input, interp_block1['out'], interp_block2['out'], interp_block3['out']], axis=3, name=self.name+'concat')
            output['out'] = res
        return output

    def inference(self, images, **kwargs):
        shape = images.get_shape()
        shape2 = tf.shape(images)
        output = {}
        if 'training' in kwargs:
            training = kwargs['training']
        else:
            training = True
        if 'pretrain' in kwargs:
            pretrain = kwargs['pretrain']
        else:
            pretrain = True
        res = self.resnet.inference(images, pretrain=pretrain, training=training)
        input_shape = res['relu4'].get_shape()[1:3]
        psp = self.build_pyramid_pooling_module(res['relu4'], input_shape, pretrain=pretrain, training=training)
        output['psp'] = psp
        input_channel = int(psp['out'].get_shape()[3])
        x = self.conv2d(self.name + 'conv5_4', psp['out'], [3, 3, input_channel, 512], use_bias=False,
                        pretrain=pretrain, train=training)
        output['conv5_4'] = x
        x = tf.nn.relu(x, name=self.name + 'relu5_4')
        output['relu1'] = x
        if 'keep_prob' not in kwargs:
            raise Exception("Keep_prob for dropout layer not given!")
        x = tf.nn.dropout(x, kwargs['keep_prob'])
        output['dropout'] = x
        out_shape = [shape2[0], int(int(shape[1]) / 4), int(int(shape[2]) / 4), 256]
        x = self.conv2d_transpose(self.name + 'deconv1', x, out_shape, [2, 2, 256, 512], stride=[1, 2, 2, 1],
                                  pretrain=pretrain, train=training)
        output['deconv1'] = x
        out_shape = [shape2[0], int(int(shape[1]) / 2), int(int(shape[2]) / 2), 128]
        x = self.conv2d_transpose(self.name + 'deconv2', x, out_shape, [2, 2, 128, 256], stride=[1, 2, 2, 1],
                                  pretrain=pretrain, train=training)
        output['deconv2'] = x
        out_shape = [shape2[0], int(shape[1]), int(shape[2]), 64]
        x = self.conv2d_transpose(self.name + 'deconv3', x, out_shape, [2, 2, 64, 128], stride=[1, 2, 2, 1],
                                  pretrain=pretrain, train=training)
        output['deconv3'] = x
        self.area = self.conv2d(self.name + 'conv6', x, [1, 1, 64, 1], pretrain=pretrain, train=training)
        output['conv6'] = self.area

        hed = self.hed.inference(res['relu2'], pretrain=pretrain, training=training)
        self.edge = hed['out']
        concat = tf.concat([self.area, self.edge], axis=3, name=self.name+'concat')
        output['concat'] = concat

        conv = self.conv2d(self.name+'add_conv0', concat, [3,3,2,32], pretrain=pretrain, train=training)
        output['add_conv0'] = conv
        relu = tf.nn.relu(conv, name=self.name+'add_relu0')
        output['add_relu0'] = relu
        conv = self.conv2d(self.name+'add_conv1', relu, [3,3,32,32], pretrain=pretrain, train=training)
        output['add_conv1'] = conv
        relu = tf.nn.relu(conv, name=self.name+'add_relu1')
        output['add_relu1'] = relu
        conv2 = self.conv2d(self.name+'add_conv2', relu, [2,2,32,64], stride=[1,2,2,1], padding='VALID', pretrain=pretrain, train=training)
        output['add_conv2'] = conv2
        relu2 = tf.nn.relu(conv2, name=self.name+'add_relu2')
        output['add_relu2'] = relu2
        conv2 = self.conv2d(self.name+'add_conv3', relu2, [3,3,64,64], pretrain=pretrain, train=training)
        output['add_conv3'] = conv2
        relu2 = tf.nn.relu(conv2, name=self.name+'add_relu2')
        output['add_relu2'] = relu2
        deconv0 = self.conv2d_transpose(self.name+'add_deconv0', relu2, [shape2[0], shape2[1], shape2[2], 32], [2,2,32,64], stride=[1,2,2,1],
                                        pretrain=pretrain, train=training)
        output['add_deconv0'] = deconv0
        relu3 = tf.nn.relu(deconv0, name=self.name+'add_relu3')
        output['add_relu3'] = relu3
        concat2 = tf.concat([relu3, relu], axis=3, name=self.name+'concat2')
        self.last_conv = self.conv2d(self.name+'last_conv', concat2, [1,1,64,1], pretrain=pretrain, train=training)
        output['last_conv'] = self.last_conv
        sigm = tf.nn.sigmoid(self.last_conv, name=self.name+'sigm')
        output['out'] = sigm

        self.pretrained_collection += self.resnet.pretrained_collection
        self.trainable_collection += self.resnet.trainable_collection
        self.all_collection += self.resnet.all_collection
        self.pretrained_collection += self.hed.pretrained_collection
        self.trainable_collection += self.hed.trainable_collection
        self.all_collection += self.hed.all_collection
        return output

    def loss(self, predicts, labels, eval_names):
        dilated_label = tf.nn.dilation2d(labels, tf.zeros([3,3,1]), strides=[1,1,1,1], rates=[1,1,1,1], padding='SAME')
        edge_label = dilated_label-labels
        hed_loss, hed_evals = self.hed.loss(self.edge, edge_label, eval_names)
        weights = labels*self.wtrue+self.wfalse
        area_loss = tf.losses.sigmoid_cross_entropy(labels, self.area, weights=weights)
        final_loss = tf.losses.sigmoid_cross_entropy(labels, self.last_conv, weights=weights)
        loss = hed_loss+area_loss+final_loss

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
            if 'area' in eval_names:
                evals['area'] = area_loss
            if 'edge' in eval_names:
                evals['edge'] = hed_loss
        return loss, evals
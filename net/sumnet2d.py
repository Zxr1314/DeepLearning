'''
sum net 2-D
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from net.channet2d import ChanNet2D
from net.pspnet2d import PSPnet2D2
from net.net import Net
import copy

class SumNet2D(Net):
    def __init__(self, common_params, net_params, name=None):
        super(SumNet2D, self).__init__(common_params, net_params, name)
        if 'threshold' in net_params:
            self.threshold = net_params['threshold']
        else:
            self.threshold = 0.1
        if 'basic_net' in net_params:
            self.basenet = net_params['basic_net']
        else:
            self.basenet = ChanNet2D(common_params, net_params, name='channet')
        if 'add_net' in net_params:
            self.addnet = net_params['add_net']
        else:
            self.addnet = PSPnet2D2(common_params, net_params, name='addnet')
        return

    def inference(self, images, **kwargs):
        output = {}
        self.add = self.addnet.inference(images, **kwargs)
        temp_args = kwargs
        temp_args['training'] = False
        temp_args['pretrain'] = True
        self.base = self.basenet.inference(images, **temp_args)
        output['base'] = self.base
        self.out = self.base['out'] + self.add['out']
        output['out'] = self.out

        self.pretrained_collection += self.basenet.pretrained_collection
        self.pretrained_collection += self.addnet.pretrained_collection
        self.trainable_collection += self.basenet.trainable_collection
        self.trainable_collection += self.addnet.trainable_collection
        self.all_collection += self.basenet.all_collection
        self.all_collection += self.addnet.all_collection
        return output

    def loss(self, predicts, labels, eval_names, weight=None):
        dif = labels-self.base['out']
        seg = tf.cast(dif>self.threshold, tf.float32)
        pre = self.addnet.last_conv
        if weight is None:
            weight = seg*self.wtrue+self.wfalse
        loss = tf.losses.sigmoid_cross_entropy(seg, pre, weights=weight)

        evals = {}
        if eval_names is not None:
            seg2 = tf.round(pre)
            if 'accuracy' in eval_names:
                evals['accuracy'] = tf.reduce_mean(tf.cast(tf.equal(seg, labels), tf.float32))
            TP = tf.cast(tf.count_nonzero(seg2 * seg), dtype=tf.float32)
            FP = tf.cast(tf.count_nonzero((1 - seg2) * seg), dtype=tf.float32)
            FN = tf.cast(tf.count_nonzero(seg2 * (1 - seg)), dtype=tf.float32)
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

'''
channet combine 2-D
'''

import tensorflow as tf

from channet2d import ChanNet2D
from resnet2d import ResNet2D4
from net import Net

class ChanComb2D(Net):
    def __init__(self, common_params, net_params, name=None):
        super(ChanComb2D, self).__init__(common_params, net_params, name)
        self.arcchan = ChanNet2D(common_params, net_params, name='arcchan')
        self.descchan = ChanNet2D(common_params, net_params, name='descchan')
        net_params['layers'] = 101
        self.resnet = ResNet2D4(common_params, net_params, name='resnet')
        self.pretrains = []
        return

    def inference(self, images, **kwargs):
        output = {}
        temp_args = kwargs
        temp_args['pretrain'] = True
        if 'training' in  kwargs:
            self.training = kwargs['training']
        else:
            self.training = True
        if 'pretrain' in kwargs:
            self.pretrain = kwargs['pretrain']
        else:
            self.pretrain = False
        if 'former_train' in kwargs:
            temp_args['training'] = kwargs['former_train']
        else:
            temp_args['training'] = False
        self.achan = self.arcchan.inference(images, **temp_args)
        output['achan'] = self.achan
        self.dchan = self.descchan.inference(images, **temp_args)
        output['dchan'] = self.dchan
        cat = tf.concat([images, self.achan['out'], self.dchan['out']], axis=3, name=self.name+'concat')
        output['cat'] = cat
        kwargs['training'] = self.training
        kwargs['pretrain'] = self.pretrain
        self.res = self.resnet.inference(cat, **kwargs)
        output['res'] = self.res
        #self.res2 = tf.reshape(self.res['out'], [tf.shape(images)[0], 1], name=self.name+'reshape')
        self.out = self.achan['out']*self.res['out']+self.dchan['out']*(1-self.res['out'])
        output['out'] = self.out

        self.pretrained_collection += self.arcchan.pretrained_collection
        self.pretrained_collection += self.descchan.pretrained_collection
        self.pretrained_collection += self.resnet.pretrained_collection
        self.trainable_collection += self.arcchan.trainable_collection
        self.trainable_collection += self.descchan.trainable_collection
        self.trainable_collection += self.resnet.trainable_collection
        self.all_collection += self.arcchan.all_collection
        self.all_collection += self.descchan.all_collection
        self.all_collection += self.resnet.all_collection
        self.pretrains.append(self.arcchan.pretrained_collection)
        self.pretrains.append(self.descchan.pretrained_collection)
        return output

    def loss(self, predicts, labels, eval_names, weight=None):
        dilated_label = tf.nn.dilation2d(labels, tf.zeros([3, 3, 1]), strides=[1, 1, 1, 1], rates=[1, 1, 1, 1],
                                         padding='SAME')
        erosed_label = tf.nn.erosion2d(labels, tf.zeros([3, 3, 1]), strides=[1, 1, 1, 1], rates=[1, 1, 1, 1],
                                       padding='SAME')
        if weight is None:
            weight = dilated_label * self.wtrue + self.wfalse + (dilated_label - erosed_label) * self.wtrue
        loss = tf.losses.mean_squared_error(labels, self.out, weight)

        evals = {}
        if eval_names is not None:
            seg = tf.round(self.out)
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

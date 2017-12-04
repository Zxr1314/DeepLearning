'''
2D U-net
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from net.net import Net

class Unet2D(Net):
    def __init__(self, common_params, net_params):
        '''

        :param common_params:
        :param net_params:
        '''
        super(Unet2D, self).__init__(common_params, net_params)
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

        return

    def inference(self, images):
        '''

        :param images:
        :return:
        '''
        # 572*572
        conv1 = self.conv2d('conv1', images, [3,3,1,64])
        relu1 = tf.nn.relu(conv1, name='relu1')
        # 570*570
        conv2 = self.conv2d('conv2', relu1, [3,3,64,64])
        relu2 = tf.nn.relu(conv2, name='relu2')
        # 568*568
        downsample1 = self.conv2d('downsample1', relu2, [2,2,64,64], stride=[1,2,2,1])
        relu3 = tf.nn.relu(downsample1, name='relu1')
        # 284*284
        conv3 = self.conv2d('conv3', relu3, [3,3,64,128])
        relu4 = tf.nn.relu(conv3, name='relu4')
        # 282*282
        conv4 = self.conv2d('conv4', relu4, [3,3,128,128])
        relu5 = tf.nn.relu(conv4, name='relu5')
        # 280*280
        downsample2 = self.conv2d('downsample2', relu5, [2,2,128,128], stride=[1,2,2,1])
        relu6 = tf.nn.relu(downsample2, name='relu6')
        # 140*140
        conv5 = self.conv2d('conv5', relu6, [3,3,128,256])
        relu7 = tf.nn.relu(conv5, name='relu7')
        # 138*138
        conv6 = self.conv2d('conv6', relu7, [3,3,256,256])
        relu8 = tf.nn.relu(conv6, name='relu8')
        # 136*136
        downsample3 = self.conv2d('downsample3', relu8, [2,2,256,256], stride=[1,2,2,1])
        relu9 = tf.nn.relu(downsample3, name='relu9')
        # 68*68
        conv7 = self.conv2d('conv7', relu9, [3,3,256,512])
        relu10 = tf.nn.relu(conv7, name='relu10')
        # 66*66
        conv8 = self.conv2d('conv8', relu10, [3,3,512,512])
        relu11 = tf.nn.relu(conv8, name='relu11')
        #64*64
        downsample4 = self.conv2d('downsample4', relu11, [3,3,512,512], stride=[1,2,2,1])
        relu12 = tf.nn.relu(downsample4, name='relu12')
        # 32*32
        conv9 = self.conv2d('conv9', relu12, [3,3,512,1024])
        relu13 = tf.nn.relu(conv9, name='relu13')
        # 30*30
        conv10 = self.conv2d('conv10', relu13, [3,3,1024,1024])
        relu14 = tf.nn.relu(conv10, name='relu14')
        # 28*28
        upsample1 = self.conv2d_transpose('upsample1', relu14, tf.shape(conv8), [2,2,512,1024],stride=[1,2,2,1])
        relu15 = tf.nn.relu(upsample1, name='relu15')
        # 56*56
        try:
            concat1 = tf.concat([relu11, relu15], axis=3, name='concat1')
        except:
            concat1 = tf.concat_v2([relu11, relu15], axis=3, name='concat1')
        conv11 = self.conv2d('conv11', concat1, [3,3,1024,512])
        relu16 = tf.nn.relu(conv11, name='relu16')
        # 54*54
        conv12 = self.conv2d('conv12', relu16, [3,3,512,512])
        relu17 = tf.nn.relu(conv12, name='relu17')
        # 52*52
        upsample2 = self.conv2d_transpose('upsample2', relu17, tf.shape(conv6), [2,2,256,512], stride=[1,2,2,1])
        relu18 = tf.nn.relu(upsample2, name='relu18')
        #104*104
        try:
            concat2 = tf.concat([relu8, relu18], axis=3, name='concat2')
        except:
            concat2 = tf.concat_v2([relu8, relu18], axis=3, name='concat2')
        conv13 = self.conv2d('conv13', concat2, [3,3,512,256])
        relu19 = tf.nn.relu(conv13, name='relu19')
        # 102*102
        conv14 = self.conv2d('conv14', relu19, [3,3,256,256])
        relu20 = tf.nn.relu(conv14, name='relu20')
        # 100*100
        upsample3 = self.conv2d_transpose('upsample3', relu20, tf.shape(conv4), [2,2,128,256], stride=[1,2,2,1])
        relu21 = tf.nn.relu(upsample3, name='relu21')
        # 200*200
        try:
            concat3 = tf.concat([relu5, relu21], axis=3, name='concat3')
        except:
            concat3 = tf.concat_v2([relu5, relu21], axis=3, name='concat3')
        conv15 = self.conv2d('conv15', concat3, [3,3,256,128])
        relu22 = tf.nn.relu(conv15, name='relu22')
        # 198*198
        conv16 = self.conv2d('conv16', relu22, [3,3,128,128])
        relu23 = tf.nn.relu(conv16, name='relu23')
        #196*196
        upsample4 = self.conv2d_transpose('upsample4', relu23, tf.shape(conv2), [2,2,64,128], stride=[1,2,2,1])
        relu24 = tf.nn.relu(upsample4, name='relu24')
        # 392*392
        try:
            concat4 = tf.concat([relu2, relu24], axis=3, name='concat4')
        except:
            concat4 = tf.concat_v2([relu2, relu24], axis=3, name='concat4')
        conv17 = self.conv2d('conv17', concat4, [3,3,128,64])
        relu25 = tf.nn.relu(conv17, name='relu25')
        # 390*390
        conv18 = self.conv2d('conv18', relu25, [3,3,64,64])
        relu26 = tf.nn.relu(conv18, name='relu26')
        # 388*388
        conv19 = self.conv2d('conv19', relu26, [1,1,64,1])

        sigm = tf.nn.sigmoid(conv19, 'sigmoid')
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

class UnetLReLU2D(Net):
    def __init__(self, common_params, net_params):
        '''

        :param common_params:
        :param net_params:
        '''
        super(UnetLReLU2D, self).__init__(common_params, net_params)
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

        return

    def inference(self, images):
        '''

        :param images:
        :return:
        '''
        # 572*572
        conv1 = self.conv2d('conv1', images, [3,3,1,64])
        relu1 = self.leaky_relu(conv1, 0.2, name='relu1')
        # 570*570
        conv2 = self.conv2d('conv2', relu1, [3,3,64,64])
        relu2 = self.leaky_relu(conv2, name='relu2')
        # 568*568
        downsample1 = self.conv2d('downsample1', relu2, [2,2,64,64], stride=[1,2,2,1])
        relu3 = self.leaky_relu(downsample1, name='relu1')
        # 284*284
        conv3 = self.conv2d('conv3', relu3, [3,3,64,128])
        relu4 = self.leaky_relu(conv3, name='relu4')
        # 282*282
        conv4 = self.conv2d('conv4', relu4, [3,3,128,128])
        relu5 = self.leaky_relu(conv4, name='relu5')
        # 280*280
        downsample2 = self.conv2d('downsample2', relu5, [2,2,128,128], stride=[1,2,2,1])
        relu6 = self.leaky_relu(downsample2, name='relu6')
        # 140*140
        conv5 = self.conv2d('conv5', relu6, [3,3,128,256])
        relu7 = self.leaky_relu(conv5, name='relu7')
        # 138*138
        conv6 = self.conv2d('conv6', relu7, [3,3,256,256])
        relu8 = self.leaky_relu(conv6, name='relu8')
        # 136*136
        downsample3 = self.conv2d('downsample3', relu8, [2,2,256,256], stride=[1,2,2,1])
        relu9 = self.leaky_relu(downsample3, name='relu9')
        # 68*68
        conv7 = self.conv2d('conv7', relu9, [3,3,256,512])
        relu10 = self.leaky_relu(conv7, name='relu10')
        # 66*66
        conv8 = self.conv2d('conv8', relu10, [3,3,512,512])
        relu11 = self.leaky_relu(conv8, name='relu11')
        #64*64
        downsample4 = self.conv2d('downsample4', relu11, [3,3,512,512], stride=[1,2,2,1])
        relu12 = self.leaky_relu(downsample4, name='relu12')
        # 32*32
        conv9 = self.conv2d('conv9', relu12, [3,3,512,1024])
        relu13 = self.leaky_relu(conv9, name='relu13')
        # 30*30
        conv10 = self.conv2d('conv10', relu13, [3,3,1024,1024])
        relu14 = self.leaky_relu(conv10, name='relu14')
        # 28*28
        upsample1 = self.conv2d_transpose('upsample1', relu14, tf.shape(conv8), [2,2,512,1024],stride=[1,2,2,1])
        relu15 = self.leaky_relu(upsample1, name='relu15')
        # 56*56
        try:
            concat1 = tf.concat([relu11, relu15], axis=3, name='concat1')
        except:
            concat1 = tf.concat_v2([relu11, relu15], axis=3, name='concat1')
        conv11 = self.conv2d('conv11', concat1, [3,3,1024,512])
        relu16 = self.leaky_relu(conv11, name='relu16')
        # 54*54
        conv12 = self.conv2d('conv12', relu16, [3,3,512,512])
        relu17 = self.leaky_relu(conv12, name='relu17')
        # 52*52
        upsample2 = self.conv2d_transpose('upsample2', relu17, tf.shape(conv6), [2,2,256,512], stride=[1,2,2,1])
        relu18 = self.leaky_relu(upsample2, name='relu18')
        #104*104
        try:
            concat2 = tf.concat([relu8, relu18], axis=3, name='concat2')
        except:
            concat2 = tf.concat_v2([relu8, relu18], axis=3, name='concat2')
        conv13 = self.conv2d('conv13', concat2, [3,3,512,256])
        relu19 = self.leaky_relu(conv13, name='relu19')
        # 102*102
        conv14 = self.conv2d('conv14', relu19, [3,3,256,256])
        relu20 = self.leaky_relu(conv14, name='relu20')
        # 100*100
        upsample3 = self.conv2d_transpose('upsample3', relu20, tf.shape(conv4), [2,2,128,256], stride=[1,2,2,1])
        relu21 = self.leaky_relu(upsample3, name='relu21')
        # 200*200
        try:
            concat3 = tf.concat([relu5, relu21], axis=3, name='concat3')
        except:
            concat3 = tf.concat_v2([relu5, relu21], axis=3, name='concat3')
        conv15 = self.conv2d('conv15', concat3, [3,3,256,128])
        relu22 = self.leaky_relu(conv15, name='relu22')
        # 198*198
        conv16 = self.conv2d('conv16', relu22, [3,3,128,128])
        relu23 = self.leaky_relu(conv16, name='relu23')
        #196*196
        upsample4 = self.conv2d_transpose('upsample4', relu23, tf.shape(conv2), [2,2,64,128], stride=[1,2,2,1])
        relu24 = self.leaky_relu(upsample4, name='relu24')
        # 392*392
        try:
            concat4 = tf.concat([relu2, relu24], axis=3, name='concat4')
        except:
            concat4 = tf.concat_v2([relu2, relu24], axis=3, name='concat4')
        conv17 = self.conv2d('conv17', concat4, [3,3,128,64])
        relu25 = self.leaky_relu(conv17, name='relu25')
        # 390*390
        conv18 = self.conv2d('conv18', relu25, [3,3,64,64])
        relu26 = self.leaky_relu(conv18, name='relu26')
        # 388*388
        conv19 = self.conv2d('conv19', relu26, [1,1,64,1])

        sigm = tf.nn.sigmoid(conv19, 'sigmoid')
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

class UnetSeLU2D(Net):
    def __init__(self, common_params, net_params):
        '''

        :param common_params:
        :param net_params:
        '''
        super(UnetSeLU2D, self).__init__(common_params, net_params)
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

        return

    def inference(self, images):
        '''

        :param images:
        :return:
        '''
        # 572*572
        conv1 = self.conv2d('conv1', images, [3,3,1,64])
        relu1 = self.selu(conv1, name='relu1')
        # 570*570
        conv2 = self.conv2d('conv2', relu1, [3,3,64,64])
        relu2 = self.selu(conv2, name='relu2')
        # 568*568
        downsample1 = self.conv2d('downsample1', relu2, [2,2,64,64], stride=[1,2,2,1])
        relu3 = self.selu(downsample1, name='relu1')
        # 284*284
        conv3 = self.conv2d('conv3', relu3, [3,3,64,128])
        relu4 = self.selu(conv3, name='relu4')
        # 282*282
        conv4 = self.conv2d('conv4', relu4, [3,3,128,128])
        relu5 = self.selu(conv4, name='relu5')
        # 280*280
        downsample2 = self.conv2d('downsample2', relu5, [2,2,128,128], stride=[1,2,2,1])
        relu6 = self.selu(downsample2, name='relu6')
        # 140*140
        conv5 = self.conv2d('conv5', relu6, [3,3,128,256])
        relu7 = self.selu(conv5, name='relu7')
        # 138*138
        conv6 = self.conv2d('conv6', relu7, [3,3,256,256])
        relu8 = self.selu(conv6, name='relu8')
        # 136*136
        downsample3 = self.conv2d('downsample3', relu8, [2,2,256,256], stride=[1,2,2,1])
        relu9 = self.selu(downsample3, name='relu9')
        # 68*68
        conv7 = self.conv2d('conv7', relu9, [3,3,256,512])
        relu10 = self.selu(conv7, name='relu10')
        # 66*66
        conv8 = self.conv2d('conv8', relu10, [3,3,512,512])
        relu11 = self.selu(conv8, name='relu11')
        #64*64
        downsample4 = self.conv2d('downsample4', relu11, [3,3,512,512], stride=[1,2,2,1])
        relu12 = self.selu(downsample4, name='relu12')
        # 32*32
        conv9 = self.conv2d('conv9', relu12, [3,3,512,1024])
        relu13 = self.selu(conv9, name='relu13')
        # 30*30
        conv10 = self.conv2d('conv10', relu13, [3,3,1024,1024])
        relu14 = self.selu(conv10, name='relu14')
        # 28*28
        upsample1 = self.conv2d_transpose('upsample1', relu14, tf.shape(conv8), [2,2,512,1024],stride=[1,2,2,1])
        relu15 = self.selu(upsample1, name='relu15')
        # 56*56
        try:
            concat1 = tf.concat([relu11, relu15], axis=3, name='concat1')
        except:
            concat1 = tf.concat_v2([relu11, relu15], axis=3, name='concat1')
        conv11 = self.conv2d('conv11', concat1, [3,3,1024,512])
        relu16 = self.selu(conv11, name='relu16')
        # 54*54
        conv12 = self.conv2d('conv12', relu16, [3,3,512,512])
        relu17 = self.selu(conv12, name='relu17')
        # 52*52
        upsample2 = self.conv2d_transpose('upsample2', relu17, tf.shape(conv6), [2,2,256,512], stride=[1,2,2,1])
        relu18 = self.selu(upsample2, name='relu18')
        #104*104
        try:
            concat2 = tf.concat([relu8, relu18], axis=3, name='concat2')
        except:
            concat2 = tf.concat_v2([relu8, relu18], axis=3, name='concat2')
        conv13 = self.conv2d('conv13', concat2, [3,3,512,256])
        relu19 = self.selu(conv13, name='relu19')
        # 102*102
        conv14 = self.conv2d('conv14', relu19, [3,3,256,256])
        relu20 = self.selu(conv14, name='relu20')
        # 100*100
        upsample3 = self.conv2d_transpose('upsample3', relu20, tf.shape(conv4), [2,2,128,256], stride=[1,2,2,1])
        relu21 = self.selu(upsample3, name='relu21')
        # 200*200
        try:
            concat3 = tf.concat([relu5, relu21], axis=3, name='concat3')
        except:
            concat3 = tf.concat_v2([relu5, relu21], axis=3, name='concat3')
        conv15 = self.conv2d('conv15', concat3, [3,3,256,128])
        relu22 = self.selu(conv15, name='relu22')
        # 198*198
        conv16 = self.conv2d('conv16', relu22, [3,3,128,128])
        relu23 = self.selu(conv16, name='relu23')
        #196*196
        upsample4 = self.conv2d_transpose('upsample4', relu23, tf.shape(conv2), [2,2,64,128], stride=[1,2,2,1])
        relu24 = self.selu(upsample4, name='relu24')
        # 392*392
        try:
            concat4 = tf.concat([relu2, relu24], axis=3, name='concat4')
        except:
            concat4 = tf.concat_v2([relu2, relu24], axis=3, name='concat4')
        conv17 = self.conv2d('conv17', concat4, [3,3,128,64])
        relu25 = self.selu(conv17, name='relu25')
        # 390*390
        conv18 = self.conv2d('conv18', relu25, [3,3,64,64])
        relu26 = self.selu(conv18, name='relu26')
        # 388*388
        conv19 = self.conv2d('conv19', relu26, [1,1,64,1])

        sigm = tf.nn.sigmoid(conv19, 'sigmoid')
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

class UnetSwish2D(Net):
    def __init__(self, common_params, net_params):
        '''

        :param common_params:
        :param net_params:
        '''
        super(UnetSwish2D, self).__init__(common_params, net_params)
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

        return

    def inference(self, images):
        '''

        :param images:
        :return:
        '''
        # 572*572
        conv1 = self.conv2d('conv1', images, [3,3,1,64])
        relu1 = self.swish(conv1, name='relu1')
        # 570*570
        conv2 = self.conv2d('conv2', relu1, [3,3,64,64])
        relu2 = self.swish(conv2, name='relu2')
        # 568*568
        downsample1 = self.conv2d('downsample1', relu2, [2,2,64,64], stride=[1,2,2,1])
        relu3 = self.swish(downsample1, name='relu1')
        # 284*284
        conv3 = self.conv2d('conv3', relu3, [3,3,64,128])
        relu4 = self.swish(conv3, name='relu4')
        # 282*282
        conv4 = self.conv2d('conv4', relu4, [3,3,128,128])
        relu5 = self.swish(conv4, name='relu5')
        # 280*280
        downsample2 = self.conv2d('downsample2', relu5, [2,2,128,128], stride=[1,2,2,1])
        relu6 = self.swish(downsample2, name='relu6')
        # 140*140
        conv5 = self.conv2d('conv5', relu6, [3,3,128,256])
        relu7 = self.swish(conv5, name='relu7')
        # 138*138
        conv6 = self.conv2d('conv6', relu7, [3,3,256,256])
        relu8 = self.swish(conv6, name='relu8')
        # 136*136
        downsample3 = self.conv2d('downsample3', relu8, [2,2,256,256], stride=[1,2,2,1])
        relu9 = self.swish(downsample3, name='relu9')
        # 68*68
        conv7 = self.conv2d('conv7', relu9, [3,3,256,512])
        relu10 = self.swish(conv7, name='relu10')
        # 66*66
        conv8 = self.conv2d('conv8', relu10, [3,3,512,512])
        relu11 = self.swish(conv8, name='relu11')
        #64*64
        downsample4 = self.conv2d('downsample4', relu11, [3,3,512,512], stride=[1,2,2,1])
        relu12 = self.swish(downsample4, name='relu12')
        # 32*32
        conv9 = self.conv2d('conv9', relu12, [3,3,512,1024])
        relu13 = self.swish(conv9, name='relu13')
        # 30*30
        conv10 = self.conv2d('conv10', relu13, [3,3,1024,1024])
        relu14 = self.swish(conv10, name='relu14')
        # 28*28
        upsample1 = self.conv2d_transpose('upsample1', relu14, tf.shape(conv8), [2,2,512,1024],stride=[1,2,2,1])
        relu15 = self.swish(upsample1, name='relu15')
        # 56*56
        try:
            concat1 = tf.concat([relu11, relu15], axis=3, name='concat1')
        except:
            concat1 = tf.concat_v2([relu11, relu15], axis=3, name='concat1')
        conv11 = self.conv2d('conv11', concat1, [3,3,1024,512])
        relu16 = self.swish(conv11, name='relu16')
        # 54*54
        conv12 = self.conv2d('conv12', relu16, [3,3,512,512])
        relu17 = self.swish(conv12, name='relu17')
        # 52*52
        upsample2 = self.conv2d_transpose('upsample2', relu17, tf.shape(conv6), [2,2,256,512], stride=[1,2,2,1])
        relu18 = self.swish(upsample2, name='relu18')
        #104*104
        try:
            concat2 = tf.concat([relu8, relu18], axis=3, name='concat2')
        except:
            concat2 = tf.concat_v2([relu8, relu18], axis=3, name='concat2')
        conv13 = self.conv2d('conv13', concat2, [3,3,512,256])
        relu19 = self.swish(conv13, name='relu19')
        # 102*102
        conv14 = self.conv2d('conv14', relu19, [3,3,256,256])
        relu20 = self.swish(conv14, name='relu20')
        # 100*100
        upsample3 = self.conv2d_transpose('upsample3', relu20, tf.shape(conv4), [2,2,128,256], stride=[1,2,2,1])
        relu21 = self.swish(upsample3, name='relu21')
        # 200*200
        try:
            concat3 = tf.concat([relu5, relu21], axis=3, name='concat3')
        except:
            concat3 = tf.concat_v2([relu5, relu21], axis=3, name='concat3')
        conv15 = self.conv2d('conv15', concat3, [3,3,256,128])
        relu22 = self.swish(conv15, name='relu22')
        # 198*198
        conv16 = self.conv2d('conv16', relu22, [3,3,128,128])
        relu23 = self.swish(conv16, name='relu23')
        #196*196
        upsample4 = self.conv2d_transpose('upsample4', relu23, tf.shape(conv2), [2,2,64,128], stride=[1,2,2,1])
        relu24 = self.swish(upsample4, name='relu24')
        # 392*392
        try:
            concat4 = tf.concat([relu2, relu24], axis=3, name='concat4')
        except:
            concat4 = tf.concat_v2([relu2, relu24], axis=3, name='concat4')
        conv17 = self.conv2d('conv17', concat4, [3,3,128,64])
        relu25 = self.swish(conv17, name='relu25')
        # 390*390
        conv18 = self.conv2d('conv18', relu25, [3,3,64,64])
        relu26 = self.swish(conv18, name='relu26')
        # 388*388
        conv19 = self.conv2d('conv19', relu26, [1,1,64,1])

        sigm = tf.nn.sigmoid(conv19, 'sigmoid')
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
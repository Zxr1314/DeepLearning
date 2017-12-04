'''
GAN 2-D
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class GAN(object):
    def __init__(self, common_params, net_params):
        '''

        :param common_params:
        :param net_params:
        '''
        self.pretrained_collection = []
        self.trainable_collection = []
        self.generator_pretrain = []
        self.discriminator_pretrain = []
        self.generator_trainable = []
        self.discriminator_trainable = []
        return

    def inference(self, image, label):
        '''

        :param image:
        :param label:
        :return:
        '''
        self.predicts = self.generator(image)
        self.discrims = self.discriminator(image, label)
        return self.predicts, self.discrims, self.generator_trainable, self.discriminator_trainable

    def loss(self, predicts, labels, discrims, eval_names):
        '''

        :param predicts:
        :param labels:
        :param discrims:
        :param eval_names:
        :return:
        '''
        raise NotImplementedError

    def generator(self, image):
        '''

        :param image:
        :return:
        '''
        raise NotImplementedError

    def discriminator(self, image, label):
        '''

        :param image:
        :param label:
        :return:
        '''
        raise NotImplementedError
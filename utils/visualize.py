from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from datetime import datetime

import tensorflow as tf
import numpy as np
from PIL import Image

class Visualize2D(object):
    '''2-D model solver
    '''
    def __init__(self, net, common_params, visualize_params):
        '''

        :param dataset:
        :param net:
        :param common_params:
        :param solver_params:
        '''
        self.batch_size = int(common_params['batch_size'])
        self.width = common_params['width']
        self.height = common_params['height']
        self.channel = int(common_params['channel'])
        if 'pretrain_model_path' in visualize_params:
            self.pretrain_path = visualize_params['pretrain_model_path']
        else:
            self.pretrain_path = 'None'
        if 'net_input' in visualize_params:
            self.net_input = visualize_params['net_input']
        else:
            self.net_input = {}
        self.save_path = visualize_params['save_path']

        self.net = net

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

        self.construct_graph()
        return

    def construct_graph(self):
        self.global_step = tf.Variable(0, trainable=False)
        self.images = tf.placeholder(tf.float32, [None, self.height, self.width, self.channel])
        self.keep_prob_holder = tf.placeholder(tf.float32)
        self.net_input['keep_prob'] = self.keep_prob_holder

        self.predicts = self.net.inference(self.images, **self.net_input)

    def initialize(self):
        #saver = tf.train.Saver()

        try:
            init = tf.global_variables_initializer()
        except:
            init = tf.initialize_all_variables()

        self.sess = tf.Session(config=self.config)

        self.sess.run(init)
        if self.pretrain_path != 'None':
            saver = tf.train.Saver(self.net.pretrained_collection, write_version=1)
            saver.restore(self.sess, self.pretrain_path)

    def visualize(self, image):
        '''

        :param input:
        :return:
        '''
        image.shape = [1, self.height, self.width, 1]
        names = []
        values = []
        for key, value in self.predicts.items():
            if type(value)==dict:
                name = key+'/'
                for key2, value2 in value.items():
                    if type(value2)==dict:
                        continue
                    names.append(name+key2)
                    values.append(value2)
            else:
                names.append(key)
                values.append(value)
        results = self.sess.run(values, feed_dict={self.images:image, self.keep_prob_holder:1.0})
        for i in range(len(results)):
            result = results[i]
            name = names[i]
            shape = result.shape
            height = shape[1]
            width = shape[2]
            channel = shape[3]
            fmin = np.min(result)
            fmax = np.max(result)
            if not os.path.exists(self.save_path+'/'+name):
                os.makedirs(self.save_path+'/'+name)
            for j in xrange(channel):
                save_image = result[:,:,:,j]
                save_image.shape = [height, width]
                save_image = (save_image-fmin)*255/(fmax-fmin)
                img = Image.fromarray(save_image.astype(np.uint8))
                img.save(self.save_path+'/'+name+'/'+str(j)+'.jpg')
        return


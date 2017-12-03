from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import sys
import time
from datetime import datetime

import tensorflow as tf
import numpy as np

from solver.solver import Solver

class Solver2D(Solver):
    '''2-D model solver
    '''
    def __init__(self, dataset, net, common_params, solver_params):
        '''

        :param dataset:
        :param net:
        :param common_params:
        :param solver_params:
        '''
        self.learning_rate = solver_params['learning_rate']
        self.beta1 = float(solver_params['beta1'])
        self.beta2 = float(solver_params['beta2'])
        self.batch_size = int(common_params['batch_size'])
        self.width = common_params['width']
        self.height = common_params['height']
        self.channel = int(common_params['channel'])
        self.testing = common_params['testing']
        if 'pretrain_model_path' in solver_params:
            self.pretrain_path = str(solver_params['pretrain_model_path'])
        else:
            self.pretrain_path = 'None'
        self.train_dir = str(solver_params['train_dir'])
        self.max_iterators = int(solver_params['max_iterators'])
        self.eval_names = solver_params['eval_names']

        self.dataset = dataset
        self.net = net

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

        self.construct_graph()
        return

    def _train(self, lr):
        '''Train model using ADAM optimizer
        '''
        train = tf.train.AdamOptimizer(lr, self.beta1, self.beta2).minimize(self.loss, global_step=self.global_step)
        #grads = opt.compute_gradients(self.loss)
        #train = opt.apply_gradients(grads, global_step=self.global_step)
        return train

    def construct_graph(self):
        self.global_step = tf.Variable(0, trainable=False)
        self.images = tf.placeholder(tf.float32, [None, self.height, self.width, self.channel])
        self.labels = tf.placeholder(tf.float32, [None, self.height, self.width, self.channel])
        self.lr = tf.placeholder(tf.float32)

        self.predicts = self.net.inference(self.images)
        self.loss, self.evals = self.net.loss(self.predicts, self.labels, self.eval_names)

        tf.summary.scalar('loss', self.loss)
        for key, value in self.evals.items():
            tf.summary.scalar(key, value)
        self.train_op = self._train(self.lr)

    def initialize(self):
        saver = tf.train.Saver(self.net.pretrained_collection, write_version=1)

        try:
            init = tf.global_variables_initializer()
        except:
            init = tf.initialize_all_variables()

        self.sess = tf.Session(config=self.config)

        self.sess.run(init)
        if self.pretrain_path != 'None':
            saver.restore(self.sess, self.pretrain_path)

    def solve(self):
        saver = tf.train.Saver(self.net.trainable_collection, write_version=1)

        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)
        if self.testing:
            n_batch = self.dataset.get_n_test_batch()
        for step in xrange(self.max_iterators):
            start_time = time.time()
            np_images, np_labels = self.dataset.batch()
            _, loss, evals = self.sess.run([self.train_op, self.loss, self.evals], feed_dict={self.images: np_images, self.labels: np_labels, self.lr: self.learning_rate[step]})
            duration = time.time()-start_time
            assert not np.isnan(loss), 'Model diverged with loss = NaN'

            if step %10 == 0:
                examples_per_sec = self.dataset.batch_size / duration
                sec_per_batch = float(duration)
                print('%s: step %d, loss = %f (%.2f examples/sec; %.3f sec/batch)' % (datetime.now(), step, loss,
                                                                                      examples_per_sec, sec_per_batch))
                print(evals)
                sys.stdout.flush()
                summary_str = self.sess.run(summary_op, feed_dict={self.images: np_images, self.labels: np_labels})
                summary_writer.add_summary(summary_str, step)
            if step % 1000 == 999:
                saver.save(self.sess, self.train_dir + '/model_'+str(step+1).zfill(6)+'.cpkt', global_step=self.global_step)
            if self.testing:
                if step % 100 == 0:
                    for i in xrange(n_batch):
                        t_start_time = time.time()
                        t_images, t_labels = self.dataset.test_batch()
                        t_loss, t_evals = self.sess.run([self.loss, self.evals], feed_dict={self.images: t_images, self.labels: t_labels})
                        t_duration = (time.time()-t_start_time)
                        print('%s: testing %d, loss = %f (%.3f sec/batch)' % (datetime.now(), i, t_loss, t_duration))
                        print(t_evals)
        # self.sess.close()
        return

    def forward(self, input):
        '''

        :param input:
        :return:
        '''
        if len(input.shape) == 1:
            input.shape = [int(input.shape[0]/self.width/self.height/self.channel), self.width, self.height, self.channel]
        elif len(input.shape) == 3:
            input.shape = [int(input.shape[0]/self.channel), input.shape[1], input.shape[2], self.channel]
        i = 0
        predict = np.zeros(input.shape)
        while i < input.shape[0]:
            images = input[i:i+self.batch_size,:,:,:]
            predict_temp = self.sess.run([self.predicts], feed_dict={self.images: images})
            predict[i:i+self.batch_size,:,:,:] = predict_temp[0]
            i += self.batch_size
        return predict
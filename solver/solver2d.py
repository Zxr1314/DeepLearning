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
from utils.plot import Plot

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
        if self.testing:
            self.test_batch_size = common_params['test_batch_size']
        if 'pretrain_model_path' in solver_params:
            self.pretrain_path = str(solver_params['pretrain_model_path'])
        else:
            self.pretrain_path = 'None'
        self.model_name = solver_params['model_name']
        self.train_dir = str(solver_params['train_dir'])
        self.max_iterators = int(solver_params['max_iterators'])
        self.eval_names = solver_params['eval_names']
        if 'keep_prob' in solver_params:
            self.keep_prob = solver_params['keep_prob']
        else:
            self.keep_prob = 1.0

        self.dataset = dataset
        self.net = net

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

        self.construct_graph()

        self.do_plot = solver_params['plot']
        if self.do_plot:
            self.plot = Plot(solver_params['plot_params'])
        return

    def _train(self, lr):
        '''Train model using ADAM optimizer
        '''
        train = tf.train.AdamOptimizer(lr, self.beta1, self.beta2).minimize(self.loss, global_step=self.global_step, var_list=self.net.trainable_collection)
        #grads = opt.compute_gradients(self.loss)
        #train = opt.apply_gradients(grads, global_step=self.global_step)
        return train

    def construct_graph(self):
        self.global_step = tf.Variable(0, trainable=False)
        self.images = tf.placeholder(tf.float32, [None, self.height, self.width, self.channel])
        self.labels = tf.placeholder(tf.float32, [None, self.height, self.width, self.channel])
        self.lr = tf.placeholder(tf.float32)
        self.keep_prob_holder = tf.placeholder(tf.float32)

        self.predicts = self.net.inference(self.images, keep_prob=self.keep_prob_holder)
        self.loss, self.evals = self.net.loss(self.predicts['out'], self.labels, self.eval_names)

        tf.summary.scalar('loss', self.loss)
        for key, value in self.evals.items():
            tf.summary.scalar(key, value)
        self.train_op = self._train(self.lr)

    def initialize(self):
        saver = tf.train.Saver(self.net.pretrained_collection, write_version=1)
        #saver = tf.train.Saver()

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
        #saver = tf.train.Saver()

        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)
        if self.testing:
            n_batch = self.dataset.get_n_test_batch()
        for step in xrange(self.max_iterators):
            start_time = time.time()
            np_images, np_labels = self.dataset.batch()
            _, loss, evals = self.sess.run([self.train_op, self.loss, self.evals], feed_dict={self.images: np_images, self.labels: np_labels, self.lr: self.learning_rate[step], self.keep_prob_holder: self.keep_prob})
            duration = time.time()-start_time
            assert not np.isnan(loss), 'Model diverged with loss = NaN'

            if step %10 == 0:
                examples_per_sec = self.dataset.batch_size / duration
                sec_per_batch = float(duration)
                print('%s: step %d, loss = %f (%.2f examples/sec; %.3f sec/batch)' % (datetime.now(), step, loss,
                                                                                      examples_per_sec, sec_per_batch))
                print(evals)
                sys.stdout.flush()
                summary_str = self.sess.run(summary_op, feed_dict={self.images: np_images, self.labels: np_labels, self.keep_prob_holder: self.keep_prob})
                summary_writer.add_summary(summary_str, step)
                if self.do_plot:
                    self.plot.plot_train(step, loss, 0)
                    if 'precision' in self.eval_names:
                        self.plot.plot_train(step, evals['precision'], 1)
                    if 'recall' in self.eval_names:
                        self.plot.plot_train(step, evals['recall'], 2)
                    if 'f1' in self.eval_names:
                        self.plot.plot_train(step, evals['f1'], 3)
            if step % 1000 == 999:
                saver.save(self.sess, self.train_dir + '/' + self.model_name + '.cpkt', global_step=self.global_step)
                if self.do_plot:
                    self.plot.save_fig()
            if self.testing:
                if (step % 500 == 0)&(step != 0):
                    temp_eval = {}
                    for name in self.eval_names:
                        temp_eval[name] = 0.0
                    temp_eval['loss'] = 0.0
                    for i in xrange(n_batch):
                        t_start_time = time.time()
                        t_images, t_labels = self.dataset.test_batch()
                        t_loss, t_evals = self.sess.run([self.loss, self.evals], feed_dict={self.images: t_images, self.labels: t_labels, self.keep_prob_holder: 1.0})
                        t_duration = (time.time()-t_start_time)
                        print('%s: testing %d, loss = %f (%.3f sec/batch)' % (datetime.now(), i, t_loss, t_duration))
                        print(t_evals)
                        temp_eval['loss'] += t_loss
                        for name in self.eval_names:
                            temp_eval[name] += t_evals[name]
                    for key,value in temp_eval.items():
                        temp_eval[key] /= float(n_batch)
                    print('testing finished.')
                    print(temp_eval)
                    if self.do_plot:
                        self.plot.plot_test(step, temp_eval['loss'], 0)
                        if 'precision' in temp_eval:
                            self.plot.plot_test(step, temp_eval['precision'], 1)
                        if 'recall' in temp_eval:
                            self.plot.plot_test(step, temp_eval['recall'], 2)
                        if 'f1' in temp_eval:
                            self.plot.plot_test(step, temp_eval['f1'], 3)
        # self.sess.close()
        if self.do_plot:
            self.plot.save_fig()
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
            images = input[i:i+self.test_batch_size,:,:,:]
            predict_temp = self.sess.run([self.predicts['out']], feed_dict={self.images: images, self.keep_prob_holder: 1.0})
            predict[i:i+self.test_batch_size,:,:,:] = predict_temp[0]
            i += self.test_batch_size
        return predict
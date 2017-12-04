'''
solver for 2-D GAN
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from datetime import datetime

import tensorflow as tf
import numpy as np

from solver.solver2d import Solver2D

class GANSolver2D(Solver2D):
    def __init__(self, dataset, net, common_params, solver_params):
        super(GANSolver2D, self).__init__(dataset, net, common_params, solver_params)
        self.d_iterations = solver_params['discriminator_iterations']
        self.g_iterations = solver_params['generator_iterations']
        return

    def _train_gan(self, lr, d_var_list, g_var_list):
        train_d = tf.train.AdamOptimizer(learning_rate=lr, beta1=self.beta1, beta2=self.beta2).minimize(self.d_loss, var_list=d_var_list)
        train_g = tf.train.AdamOptimizer(learning_rate=lr, beta1=self.beta1, beta2=self.beta2).minimize(self.g_loss, var_list=g_var_list)
        return train_d, train_g

    def construct_graph(self):
        self.global_step = tf.Variable(0, trainable=False)
        self.images = tf.placeholder(tf.float32, [self.batch_size, self.height, self.width, self.channel])
        self.labels = tf.placeholder(tf.float32, [self.batch_size, self.height, self.width, self.channel])
        self.lr = tf.placeholder(tf.float32)

        self.predicts, self.discrims, self.g_var_list, self.d_var_list = self.net.inference(self.images, self.labels)
        self.d_loss, self.g_loss, self.evals = self.net.loss(self.predicts, self.labels, self.discrims, self.eval_names)

        tf.summary.scalar('loss', self.loss)
        for key, value in self.evals.items():
            tf.summary.scalar(key, value)
        self.train_d, self.train_g = self._train_gan(self.lr, self.d_var_list, self.g_var_list)
        return

    def solve(self):
        saver = tf.train.Saver(self.net.trainable_collection, write_version=1)

        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)
        if self.testing:
            n_batch = self.dataset.get_n_test_batch()
        for step in xrange(self.max_iterators):
            print("Start iteration %d..."%step)
            for i in xrange(self.d_iterations):
                np_images, np_labels = self.dataset.batch()
                _, d_loss = self.sess.run([self.train_d, self.d_loss], feed_dict={self.images: np_images, self.labels: np_labels})
                print("\tTrain discriminator %d: loss=%f"%(i, d_loss))
            for i in xrange(self.g_iterations):
                np_images, np_labels = self.dataset.batch()
                _, g_loss, evals = self.sess.run([self.train_g, self.g_loss, self.evals], feed_dict={self.images: np_images, self.labels: np_labels})
                print("\tTrain generator %d: loss=%f"%(i, g_loss))
                print(self.evals)
            if (step%500==0)&(step!=0):
                if self.testing:
                    temp_eval = {}
                    for name in self.eval_names:
                        temp_eval[name] = 0.0
                    temp_eval['loss'] = 0.0
                    for i in xrange(n_batch):
                        t_start_time = time.time()
                        t_images, t_labels = self.dataset.test_batch()
                        t_loss, t_evals = self.sess.run([self.loss, self.evals],
                                                        feed_dict={self.images: t_images, self.labels: t_labels})
                        t_duration = (time.time() - t_start_time)
                        print('%s: testing %d, loss = %f (%.3f sec/batch)' % (datetime.now(), i, t_loss, t_duration))
                        print(t_evals)
                        temp_eval['loss'] += t_loss
                        for name in self.eval_names:
                            temp_eval[name] += t_evals[name]
                    for key, value in temp_eval.items():
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
            if step%1000==999:
                saver.save(self.sess, self.train_dir + '/' + self.model_name + '.cpkt', global_step=self.global_step)
                if self.do_plot:
                    self.plot.save_fig()
        return
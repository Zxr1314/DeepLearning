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

    def __train(self, lr):
        train_d = tf.train.AdamOptimizer(learning_rate=lr, beta1=self.beta1, beta2=self.beta2).minimize(self.d_loss)
        train_g = tf.train.AdamOptimizer(learning_rate=lr, beta1=self.beta1, beta2=self.beta2).minimize(self.g_loss)
        return train_d, train_g

    def construct_graph(self):
        self.global_step = tf.Variable(0, trainable=False)
        self.images = tf.placeholder(tf.float32, [self.batch_size, self.height, self.width, self.channel])
        self.labels = tf.placeholder(tf.float32, [self.batch_size, self.height, self.width, self.channel])
        self.lr = tf.placeholder(tf.float32)

        self.predicts = self.net.inference(self.images)
        self.d_loss, self.g_loss, self.evals = self.net.loss(self.predicts, self.labels, self.eval_names)

        tf.summary.scalar('loss', self.loss)
        for key, value in self.evals.items():
            tf.summary.scalar(key, value)
        self.train_d, self.train_g = self._train(self.lr)
        return

    def solve(self):
        saver = tf.train.Saver(self.net.trainable_collection, write_version=1)

        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)
        if self.testing:
            n_batch = self.dataset.get_n_test_batch()
        for step in xrange(self.max_iterators):
            for i in xrange(self.d_iterations):

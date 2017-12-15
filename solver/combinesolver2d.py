

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from solver.solver2d import Solver2D
from utils.tensorboard import *

class CombineSolver2D(Solver2D):
    def __init__(self, dataset, net, common_params, solver_params):
        super(CombineSolver2D, self).__init__(dataset, net, common_params, solver_params)
        return

    def initialize(self):
        try:
            init = tf.global_variables_initializer()
        except:
            init = tf.initialize_all_variables()
        self.sess = tf.Session(config=self.config)
        self.sess.run(init)
        if self.net.pretrains==[] or type(self.net.pretrains[0]) != list:
            saver = tf.train.Saver(var_list=self.net.pretrained_collection)
            saver.restore(self.sess, self.pretrain_path)
        else:
            n_pretrain = len(self.net.pretrains)
            if n_pretrain > 1 and type(self.pretrain_path) != list:
                raise Exception("pretrain path not enough!")
            if n_pretrain > 1 and len(self.pretrain_path) != n_pretrain:
                raise Exception("pretrain path not enough!")

            for i in range(n_pretrain):
                saver = tf.train.Saver(var_list=self.net.pretrains[i])
                saver.restore(self.sess, self.pretrain_path[i])
        return


'''
ploting class
not finished
'''

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

class Plot(object):
    def __init__(self, plot_params):
        '''

        '''
        self.max_iterations = plot_params['max_iterations']
        if 'train_marker' in plot_params:
            self.train_marker = plot_params['train_marker']
        else:
            self.train_marker = '.'
        if 'train_size' in plot_params:
            self.train_size = plot_params['train_size']
        else:
            self.train_size = 9
        if 'train_color' in plot_params:
            self.train_color = plot_params['train_color']
        else:
            self.train_color = 'b'
        if 'test_marker' in plot_params:
            self.test_marker = plot_params['test_marker']
        else:
            self.test_marker = '.'
        if 'test_size' in plot_params:
            self.test_size = plot_params['test_size']
        else:
            self.test_size = 20
        if 'test_color' in plot_params:
            self.test_color = plot_params['test_color']
        else:
            self.test_color = 'r'
        self.save_name = plot_params['save_name']
        self.interactive = plot_params['interactive']
        self.figure = mpl.pyplot.figure(figsize=(18.5,10.5))
        self.loss_plt = self.figure.add_subplot(221)
        self.loss_plt.axis([0, self.max_iterations, 0, 0.3])
        self.loss_plt.set_title('loss')
        self.pre_plt = self.figure.add_subplot(222)
        self.pre_plt.axis([0, self.max_iterations, 0, 1])
        self.pre_plt.set_title('precision')
        self.rec_plt = self.figure.add_subplot(223)
        self.rec_plt.axis([0, self.max_iterations, 0, 1])
        self.rec_plt.set_title('recall')
        self.f1_plt = self.figure.add_subplot(224)
        self.f1_plt.axis([0, self.max_iterations, 0, 1])
        self.f1_plt.set_title('f1')
        if self.interactive:
            self.figure.show()
        return

    def plot_train(self, iter, value, type):
        if type==1:
            plt_name = 'pre_plt'
        elif type==2:
            plt_name = 'rec_plt'
        elif type==3:
            plt_name = 'f1_plt'
        else:
            plt_name = 'loss_plt'
        str = format("self.%s.scatter(iter, value, s=self.train_size, c=self.train_color, marker=self.train_marker)"%plt_name)
        eval(str);
        plt.pause(0.1)
        return

    def plot_test(self, iter, value, type):
        if type==1:
            plt_name = 'pre_plt'
        elif type==2:
            plt_name = 'rec_plt'
        elif type==3:
            plt_name = 'f1_plt'
        else:
            plt_name = 'loss_plt'
        str = format("self.%s.scatter(iter, value, s=self.test_size, c=self.test_color, marker=self.test_marker)"%plt_name)
        eval(str);
        plt.pause(0.1)
        return

    def save_fig(self):
        self.figure.savefig(self.save_name)
        return


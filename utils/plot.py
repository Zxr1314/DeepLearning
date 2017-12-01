'''
ploting class
not finished
'''

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
        plt.scatter()
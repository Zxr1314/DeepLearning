import numpy as np

class Augmentations(object):
    def __init__(self, atype, **kwargs):
        if atype=='scaling':
            self.func = self.scaling
        else:
            raise NotImplementedError
        self.args = kwargs
        return

    def process(self, input):
        output = self.func(input, **self.args)
        return output

    def scaling(self, input, **kwargs):
        if 'div' in kwargs:
            div = kwargs['div']
        else:
            div = np.max(input)
        if 'bias' in kwargs:
            bias = kwargs['bias']
        else:
            bias = 0
        output = input/div+bias
        return output
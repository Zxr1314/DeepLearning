'''Solver Abstract Class
'''

class Solver(object):
    def __init__(self, dataset, net, common_params, solver_params):
        raise NotImplementedError

    def solve(self):
        raise NotImplementedError

    def forward(self, input):
        raise NotImplementedError

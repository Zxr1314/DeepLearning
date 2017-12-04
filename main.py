'''
test file
'''

import time
import numpy as np

from dataset.fdataset import FDataSet
from net.unet2d import Unet2D
from solver.solver2d import Solver2D

def test_file(solve, file_name, save_name):
    data = np.fromfile(file_name, dtype=np.float32)
    stime = time.time()
    predict = solve.forward(data)
    duration = time.time()-stime
    print('testing cost %f seconds.'%duration)
    predict.astype(np.float32)
    predict.tofile(save_name)
    return predict

common_params = {}
common_params['batch_size'] = 4
common_params['test_batch_size'] = 8
common_params['dimension'] = 2
common_params['width'] = 512
common_params['height'] = 512
common_params['channel'] = 1
common_params['testing'] = True

dataset_params = {}
dataset_params['dtype'] = np.uint16
dataset_params['random'] = True
dataset_params['data_path'] = '/media/E/Documents/VesselData/TrainData/patches2D512'
dataset_params['label_path'] = '/media/E/Documents/VesselData/TrainLabel/patches2D512'
dataset_params['test_data_path'] = '/media/E/Documents/VesselData/TrainData/testpatches2D512'
dataset_params['test_label_path'] = '/media/E/Documents/VesselData/TrainLabel/testpatches2D512'

net_params = {}
net_params['weight_true'] = 8
net_params['weight_false'] = 2

solver_params = {}
solver_params['train_dir'] = 'models'
#solver_params['pretrain_model_path'] = 'models/model_076000.cpkt-76000'
solver_params['max_iterators'] = 10000
learning_rate = np.zeros(10000, dtype=np.float32)
learning_rate[0:10000] = 0.001
learning_rate[10000:40000] = 0.001
learning_rate[40000:70000] = 0.0005
learning_rate[70000:100000] = 0.0001
solver_params['learning_rate'] = learning_rate
solver_params['beta1'] = 0.9
solver_params['beta2'] = 0.999
eval_names = []
eval_names.append('accuracy')
eval_names.append('precision')
eval_names.append('recall')
eval_names.append('f1')
solver_params['eval_names'] = eval_names
plot_params = {}
plot_params['max_iterations'] = solver_params['max_iterators']
plot_params['save_name'] = 'output/relu.png'
plot_params['interactive'] = True
solver_params['plot'] = True
solver_params['plot_params'] = plot_params

dataset = FDataSet(common_params, dataset_params)
net = Unet2D(common_params, net_params)
solver = Solver2D(dataset, net, common_params, solver_params)
solver.initialize()
solver.solve()
'''test_file(solver, '/media/E/Documents/VesselData/TrainData/0005/oridata.dat',
          '/media/E/Documents/VesselData/TrainLabel/0005/unet_76000.dat')
test_file(solver, '/media/E/Documents/VesselData/TrainData/0015/oridata.dat',
          '/media/E/Documents/VesselData/TrainLabel/0015/unet_76000.dat')
test_file(solver, '/media/E/Documents/VesselData/TrainData/0115/oridata.dat',
          '/media/E/Documents/VesselData/TrainLabel/0115/unet_76000.dat')
test_file(solver, '/media/E/Documents/VesselData/TrainData/0322/oridata.dat',
          '/media/E/Documents/VesselData/TrainLabel/0322/unet_76000.dat')'''

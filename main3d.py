'''
test file
'''

import time
import numpy as np

from dataset.fdataset import FDataSet
from net.pspnet3d import *
from solver.solver3d import Solver3D

def test_file(solve, file_name, save_name):
    data = np.fromfile(file_name, dtype=np.float32)
    stime = time.time()
    predict = solve.forward(data)
    duration = time.time()-stime
    print('testing cost %f seconds.'%duration)
    predict = predict.astype(np.float32)
    predict.tofile(save_name)
    return predict

common_params = {}
common_params['batch_size'] = 1
common_params['test_batch_size'] = 4
common_params['dimension'] = 3
common_params['width'] = 128
common_params['height'] = 128
common_params['depth'] = 128
common_params['channel'] = 1
common_params['testing'] = True

dataset_params = {}
dataset_params['dtype'] = np.uint16
dataset_params['random'] = True
dataset_params['data_path'] = '/media/E/Documents/VesselData/TrainData/3D128'
dataset_params['label_path'] = '/media/E/Documents/VesselData/TrainLabel/3D128'
dataset_params['test_data_path'] = '/media/E/Documents/VesselData/TrainData/test3D128'
dataset_params['test_label_path'] = '/media/E/Documents/VesselData/TrainLabel/test3D128'

net_params = {}
net_params['weight_true'] = 4
net_params['weight_false'] = 2
net_params['layers'] = 50
solver_params = {}
solver_params['train_dir'] = 'models'
#solver_params['model_name'] = 'relu'
#solver_params['model_name'] = 'lrelu'
#solver_params['model_name'] = 'selu'
#solver_params['model_name'] = 'swish'
solver_params['model_name'] = 'pspnet3d'
#solver_params['pretrain_model_path'] = 'models/arcpsp.cpkt-30000'
solver_params['max_iterators'] = 10000
learning_rate = np.zeros(10000, dtype=np.float32)
learning_rate[0:10000] = 0.001
'''learning_rate[10000:40000] = 0.001
learning_rate[40000:70000] = 0.0005
learning_rate[70000:100000] = 0.0001'''
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
#plot_params['save_name'] = 'output/relu.png'
#plot_params['save_name'] = 'output/lrelu.png'
#plot_params['save_name'] = 'output/selu.png'
plot_params['save_name'] = 'output/pspnet3d.png'
plot_params['interactive'] = True
solver_params['plot'] = True
solver_params['plot_params'] = plot_params
solver_params['keep_prob'] = 0.9
net_input = {}
net_input['training'] = True
net_input['former_train'] = False
net_input['pretrain'] = False
solver_params['net_input'] = net_input

dataset = FDataSet(common_params, dataset_params)
#net = Unet2D(common_params, net_params)
#net = UnetLReLU2D(common_params, net_params)
#net = UnetSeLU2D(common_params, net_params)
net = PSPnet3D(common_params, net_params, name='pspnet1')
#net = PSPnet2DCombine(common_params, net_params)
#solver = CombineSolver2D(dataset, net, common_params, solver_params)
solver = Solver3D(dataset, net, common_params, solver_params)
solver.initialize()
solver.solve()
test_file(solver, '/media/E/Documents/VesselData/TrainData/0005/oridata.dat',
          '/media/E/Documents/VesselData/TrainLabel/0005/descpsp_30000.dat')
test_file(solver, '/media/E/Documents/VesselData/TrainData/0049/oridata.dat',
          '/media/E/Documents/VesselData/TrainLabel/0049/descpsp_30000.dat')
test_file(solver, '/media/E/Documents/VesselData/TrainData/0115/oridata.dat',
          '/media/E/Documents/VesselData/TrainLabel/0115/descpsp_30000.dat')
test_file(solver, '/media/E/Documents/VesselData/TrainData/0322/oridata.dat',
          '/media/E/Documents/VesselData/TrainLabel/0322/descpsp_30000.dat')

import time

from net.channet2d import *
from utils.visualize import *

common_params = {}
common_params['batch_size'] = 1
common_params['dimension'] = 2
common_params['width'] = 512
common_params['height'] = 512
common_params['channel'] = 1

net_params = {}
net_params['weight_true'] = 2
net_params['weight_false'] = 1
net_params['layers'] = 50

visualize_params = {}
visualize_params['train_dir'] = 'models'
visualize_params['model_name'] = 'channet'
visualize_params['pretrain_model_path'] = 'models/channet.cpkt-28000'
net_input = {}
net_input['training'] = True
net_input['former_train'] = True
net_input['pretrain'] = True
visualize_params['net_input'] = net_input
visualize_params['save_path'] = 'output/feature_visualize'

net = ChanNet2D(common_params, net_params, name='channet')
vis = Visualize2D(net, common_params, visualize_params)
vis.initialize()
image = np.fromfile('/media/E/Documents/VesselData/TrainData/patches2D512/00500.dat', dtype=np.uint16).astype(np.float32)
vis.visualize(image)
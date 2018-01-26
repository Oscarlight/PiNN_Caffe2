import sys
sys.path.append('../')
import numpy as np
import glob
from itertools import product
from pinn_api import DeviceModel, plot_iv
import pinn.parser as parser
import pinn.preproc as preproc
import pinn.exporter as exporter
import time
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='pinn')
parser.add_argument("model_name", type=str, default='transiXOR_Mdoels/example',
                    help="model_name")
parser.add_argument("-mls", type=float, default=1e2,
                    help="max loss scale")
parser.add_argument("-lr", type=float, default=0.1,
                    help="base learning rate")
parser.add_argument("-epoch", type=int, default=5e5,
                    help="num of epoch")
parser.add_argument("-report", type=float, default=1e3,
                    help="report_interval")
parser.add_argument("-hidden", type=int, default=8,
                    help="hidden dimension")
parser.add_argument("-layer", type=int, default=2,
                    help="number of hidden layers")
parser.add_argument("-batchsize", type=int, default=1024,
                    help="batch size")
args = parser.parse_args()

# ----------------- Train + Eval ---------------------
dc_model = DeviceModel(
	args.model_name,
	sig_input_dim=1,  # due to Vtg and Vbg are interchangeable
	tanh_input_dim=1,
	output_dim=1,
	train_target='origin',
	net_builder='origin', # use 'adjoint' here to generate adjoint net
)

## manually input the number of train/eval examples
train_example = 7938 # 56313 
test_example  = 882 # 56314 
dc_model.add_database('train', 'db/train.minidb', train_example, 'db/preproc_param.p')
dc_model.add_database('eval', 'db/eval.minidb', test_example, 'db/preproc_param.p')

dc_model.build_nets(
	hidden_sig_dims=[args.hidden] * args.layer + [1],
	hidden_tanh_dims=[args.hidden] * args.layer + [1],
	train_batch_size=args.batchsize,
	eval_batch_size=args.batchsize,
	weight_optim_method='AdaGrad',
	weight_optim_param={'alpha':args.lr, 'epsilon':1e-4},
	bias_optim_method='AdaGrad',
	bias_optim_param={'alpha':args.lr, 'epsilon':1e-4},
	loss_function='scaled_l2', # or 'scaled_l2'
	max_loss_scale=args.mls, 
)

# dc_model.draw_nets()

start = time.time()
dc_model.train_with_eval(
	num_epoch=int(args.epoch),
	report_interval=int(args.report),
	eval_during_training=True
)
end = time.time()
print('Elapsed time: ' + str(end - start))
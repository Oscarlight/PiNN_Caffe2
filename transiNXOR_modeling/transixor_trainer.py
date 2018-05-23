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
parser.add_argument("model_name", type=str, default='transiXOR_Models/example',
                    help="model_name")
parser.add_argument("-mls", type=float, default=1e2,
                    help="max loss scale")
parser.add_argument("-lr", type=float, default=0.1,
                    help="base learning rate")
parser.add_argument("-epoch", type=float, default=1e5,
                    help="num of epoch")
parser.add_argument("-report", type=float, default=1e3,
                    help="report_interval")
parser.add_argument("-hidden", nargs='+', type=int, default=[8],
                    help="hidden dimension")
parser.add_argument("-batchsize", type=int, default=1024,
                    help="batch size")
parser.add_argument("-lossfunct", type=str, default="scaled_l1",
                    help="type of loss function")
parser.add_argument("-neg_grad_mag", type=float, default=100.0,
                    help="negative gradient penalty magnitude")
args = parser.parse_args()

# ----------------- Train + Eval ---------------------
dc_model = DeviceModel(
	args.model_name,
	sig_input_dim=1,  # due to Vtg and Vbg are interchangeable
	tanh_input_dim=1,
	output_dim=1,
	train_target='origin',
	net_builder='adjoint', # use 'adjoint' here to generate adjoint net
)

## manually input the number of train/eval examples
train_example = 60516 
test_example  = 6724 
dc_model.add_database('train', 'db/train.minidb', train_example, 'db/preproc_param.p')
dc_model.add_database('eval', 'db/eval.minidb', test_example, 'db/preproc_param.p')

neg_grad_penalty = {
	'input_type': 'tanh',
	'input_idx': [0],
	'magnitude': args.neg_grad_mag,
}

init_model = {
	'name': './transiXOR_Models/bise_ext_sym_h264_0_init',
	'prefix': 'adjoint/',
}

dc_model.build_nets(
	hidden_sig_dims=args.hidden + [1],
	hidden_tanh_dims=args.hidden + [1],
	train_batch_size=args.batchsize,
	eval_batch_size=args.batchsize,
	weight_optim_method='AdaGrad',
	weight_optim_param={'alpha':args.lr, 'epsilon':1e-4},
	bias_optim_method='AdaGrad',
	bias_optim_param={'alpha':args.lr, 'epsilon':1e-4},
	loss_function=args.lossfunct,
	max_loss_scale=args.mls,
	neg_grad_penalty=neg_grad_penalty,
	init_model=init_model,
)

dc_model.draw_nets()

start = time.time()
dc_model.train_with_eval(
	num_epoch=int(args.epoch),
	report_interval=int(args.report),
	eval_during_training=True
)
end = time.time()
print('Elapsed time: ' + str(end - start))

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
parser.add_argument("model_name", type=str,
                    help="model_name")
parser.add_argument("-lr", type=float, default=0.1,
                    help="base learning rate")
parser.add_argument("-epoch", type=float,
                    help="num of epoch")
parser.add_argument("-report", type=float, default=1e3,
                    help="report_interval")
parser.add_argument("-hidden", type=int, default=32,
                    help="hidden dimension")
parser.add_argument("-batchsize", type=int, default=1024,
                    help="batch size")
parser.add_argument("-terminal", type=str,
                    help="which terminal: 'b', 'd', 'g', 's'")
args = parser.parse_args()

# ----------------- Train + Eval ---------------------
dc_model = ACQVModel(
	args.model_name,
	input_dim=4,
	output_dim=1,
)
dc_model.add_database(
	'train', args.terminal + '_train.minidb', 
	56313, args.terminal + '_preproc_param.p')
dc_model.add_database(
	'eval', args.terminal + '_eval.minidb',
	56313, args.terminal + '_preproc_param.p')

dc_model.build_nets(
	hidden_dims=[args.hidden, args.hidden],
	batch_size=args.batchsize,
	optim_method='AdaGrad',
	optim_param={'alpha':args.lr, 'epsilon':1e-4},
)

start = time.time()
dc_model.train_with_eval(
	num_epoch=int(args.epoch),
	report_interval=int(args.report),
	eval_during_training=True
)
end = time.time()
print('Elapsed time: ' + str(end - start))
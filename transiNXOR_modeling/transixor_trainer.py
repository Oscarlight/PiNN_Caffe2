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

parser = argparse.ArgumentParser(description='pinn')
parser.add_argument("model_name", type=str,
                    help="model_name")
parser.add_argument("-mls", type=float,
                    help="max loss scale")
parser.add_argument("-lr", type=float,
                    help="base learning rate")
parser.add_argument("-epoch", type=float,
                    help="num of epoch")
parser.add_argument("-report", type=float,
                    help="report_interval")
parser.add_argument("-hidden", type=int,
                    help="hidden dimension")
parser.add_argument("-batchsize", type=int,
                    help="batch size")
args = parser.parse_args()

# ----------------- Preprocessing --------------------
id_file = glob.glob('./transiXOR_data/*_id_*')
id_data = np.load(id_file[0])
# vds, vbg, vtg, id
print(id_data.shape)
vds = np.linspace(0, 0.4, 41)
vbg = np.linspace(0, 0.4, 41)
vtg = np.linspace(0, 0.4, 67)
iter_lst = list(product(vds, vbg, vtg))
vds_train = np.expand_dims(np.array([e[0] for e in iter_lst], dtype=np.float32), axis=1)
vbg_train = np.array([e[1] for e in iter_lst], dtype=np.float32)
vtg_train = np.array([e[2] for e in iter_lst], dtype=np.float32)
id_train = np.expand_dims(id_data.flatten(), axis=1).astype(np.float32)
vg_train = np.column_stack((vtg_train, vbg_train))
print(vg_train.shape)
print(vds_train.shape)
print(id_train.shape)
data_arrays = [vg_train, vds_train, id_train]

scale, vg_shift = preproc.compute_dc_meta(*data_arrays)
preproc_param = {
	'scale' : scale, 
	'vg_shift' : vg_shift, 
}
print(scale, vg_shift)
np.random.seed = 42
permu = np.random.permutation(len(data_arrays[0]))
num_eval = int(len(data_arrays[0])*0.1)
data_arrays = [e[permu] for e in data_arrays]
data_arrays_eval = [e[0:num_eval] for e in data_arrays]
data_arrays_train = [e[num_eval:] for e in data_arrays]
print(data_arrays_train[0].shape)

# ----------------- Train + Eval ---------------------
dc_model = DeviceModel(
	args.model_name,
	sig_input_dim=2,
	tanh_input_dim=1,
	output_dim=1,
	train_target='origin'
)
dc_model.add_data('train', data_arrays_train, preproc_param)
dc_model.add_data('eval',data_arrays_eval, preproc_param)
# plot_iv(*dc_model.preproc_data_arrays)

dc_model.build_nets(
	hidden_sig_dims=[args.hidden, 1],
	hidden_tanh_dims=[args.hidden, 1],
	train_batch_size=args.batchsize,
	eval_batch_size=args.batchsize,
	weight_optim_method='AdaGrad',
	weight_optim_param={'alpha':args.lr, 'epsilon':1e-4},
	bias_optim_method='AdaGrad',
	bias_optim_param={'alpha':args.lr, 'epsilon':1e-4},
	loss_function='scaled_l1', # or 'scaled_l2'
	max_loss_scale=args.mls, 
)

start = time.time()
dc_model.train_with_eval(
	num_epoch=int(args.epoch),
	report_interval=int(args.report),
	eval_during_training=True
)
end = time.time()
print('Elapsed time: ' + str(end - start))
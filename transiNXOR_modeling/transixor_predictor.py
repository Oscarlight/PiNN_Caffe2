import sys, os
sys.path.append('../')
import numpy as np
from itertools import product
from pinn_api import predict_ids_grads, predict_ids
import matplotlib.pyplot as plt
import glob

## ------------  Input  ---------------
VDS = 0.2
VTG = None
VBG = 0.2
## ------------  True data  ---------------
ids_file = glob.glob('./transiXOR_data/current_D9.npy')
data_true = np.load(ids_file[0])
## ------------  Helper Funs  ---------------
def ids(data, vds=None, vbg=None, vtg=None):
	# vds, vbg, vtg, id
	# assume v from -0.1 to 0.3, total 41 points
	vid = lambda v : int(round((v + 0.1)/0.01))
	if vds and vbg:
		return data[vid(vds), vid(vbg), :]
	if vtg and vbg:
		return data[:, vid(vbg), vid(vtg)]
	else:
		raise Exception('Not Supported')

def plot(data_pred, data_true, vds=None, vbg=None, vtg=None):
	plt.plot(ids(data_pred, vds=vds, vbg=vbg, vtg=vtg), 'r')
	plt.plot(ids(data_true, vds=vds, vbg=vbg, vtg=vtg), 'b')
	plt.show()
	plt.semilogy(ids(data_pred, vds=vds, vbg=vbg, vtg=vtg), 'r')
	plt.semilogy(ids(data_true, vds=vds, vbg=vbg, vtg=vtg), 'b')
	plt.show()
## ------------  Prediction ---------------
pred_data_path = 'pred_data/'
# model_name = 'bise_ext_sym_h264_0'
model_name = 'bise_ext_sym_h264_neggrad_0'
if not os.path.isfile(pred_data_path + model_name + '.npy'):
	print('Computing all data...')
	vds = np.linspace(-0.1, 0.3, 41)
	vbg = np.linspace(-0.1, 0.3, 41)
	vtg = np.linspace(-0.1, 0.3, 41)
	iter_lst = list(product(vds, vbg, vtg))
	vds_pred = np.expand_dims(np.array([e[0] for e in iter_lst], dtype=np.float32), axis=1)
	vbg_pred = np.array([e[1] for e in iter_lst], dtype=np.float32)
	vtg_pred = np.array([e[2] for e in iter_lst], dtype=np.float32)
	vg_pred = np.column_stack((vtg_pred, vbg_pred))
	vg_pred = np.sum(vg_pred, axis=1, keepdims=True) # model use symmetry vtg vbg
	model_path = './transiXOR_Models/'
	## If trained with adjoint builder
	data_pred_flat, _, _ = predict_ids_grads(
		model_path + model_name, vg_pred, vds_pred)
	data_pred = np.zeros((41, 41, 41))
	idx = 0
	for i in range(41):
		for j in range(41):
			for k in range(41):
				data_pred[i, j, k] = data_pred_flat[idx]
				idx += 1
	## If trained with origin builder
	# ids_pred = predict_ids(
	# 	model_path + model_name, vg_pred, vds_pred)
	np.save(pred_data_path + model_name + '.npy', data_pred) 
	plot(data_pred, data_true, vds=VDS, vbg=VBG, vtg=VTG)
else:
	print('Reading from pre-computed data...')
	data_pred = np.load(pred_data_path + model_name + '.npy')
	print(data_pred.shape)
	plot(data_pred, data_true, vds=VDS, vbg=VBG, vtg=VTG)

## Point test
# ids_pred = predict_ids(
# 	'./transiXOR_Models/bise_ext_sym_h264_0',
# 	np.array([0.2+0.2]), np.array([0.2]))
# print(ids_pred)
# ids_pred = predict_ids(
# 	'./transiXOR_Models/bise_ext_sym_h264_0',
# 	np.array([0.0+0.0]), np.array([0.2]))
# print(ids_pred)
# ids_pred = predict_ids(
# 	'./transiXOR_Models/bise_ext_sym_h264_0',
# 	np.array([0.0+0.1]), np.array([0.2]))
# print(ids_pred)
# ids_pred = predict_ids(
# 	'./transiXOR_Models/bise_ext_sym_h264_0',
# 	np.array([0.1+0.0]), np.array([0.2]))

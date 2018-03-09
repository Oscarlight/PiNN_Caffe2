import sys
sys.path.append('../')
import caffe2_paths
import numpy as np
import glob
from itertools import product
import pinn.preproc as preproc
import pinn.data_reader as data_reader
import matplotlib.pyplot as plt
import pickle
import os

# ----------------- Preprocessing --------------------
vds = np.concatenate((np.linspace(-0.1, -0.01, 10),np.linspace(0.01, 0.3, 30)))
# print(vds)
vbg = np.linspace(-0.1, 0.3, 41)
# print(vbg)
vtg = np.linspace(-0.1, 0.3, 41)
# print(vtg)
id_file = glob.glob('./transiXOR_data/current_D9.npy')

db_path = 'db/'
id_data = np.load(id_file[0])

# !!CAUTION!! If use batch direct weighted L1 loss,
#             make sure no zero label in the training data
#             Future version will address this issue internally.
# selected_vds_idx = [1, 5, 9, 12, 15, 17, 18, 19, 20]
# vds = vds[selected_vds_idx]
# id_data = id_data[selected_vds_idx,:,:]
id_data = np.concatenate((id_data[0:11,:,:],id_data[12:,:,:]))

## Check whether zero label exit
assert np.min(np.abs(id_data).flatten()) < 1e-9, "Zero exist in labels"

# vds, vbg, vtg, id
print('original data shape: ' 
	+ str(id_data.shape) + '; ' 
	+ str(id_data.shape[0] * id_data.shape[1] * id_data.shape[2])
)

iter_lst = list(product(vds, vbg, vtg))
vds_train = np.expand_dims(np.array([e[0] for e in iter_lst], dtype=np.float32), axis=1)
vbg_train = np.array([e[1] for e in iter_lst], dtype=np.float32)
vtg_train = np.array([e[2] for e in iter_lst], dtype=np.float32)
id_train = np.expand_dims(id_data.flatten(), axis=1).astype(np.float32)
vg_train = np.column_stack((vtg_train, vbg_train))
print('--- Original shape: ')
print(vg_train.shape)
print(vds_train.shape)
print(id_train.shape)

## Using the fact that vtg and vbg are interchangeable
## CAUTION: This invariance may not be true for experimental data
# vg_train = np.sum(vg_train, axis=1, keepdims=True)

## random select train/eval = 0.9/0.1
np.random.seed = 42
data_arrays = [vg_train, vds_train, id_train]
permu = np.random.permutation(len(data_arrays[0]))
num_eval = int(len(data_arrays[0])*0.1)
data_arrays = [e[permu] for e in data_arrays]
data_arrays_eval = [e[0:num_eval] for e in data_arrays]
data_arrays_train = [e[num_eval:] for e in data_arrays]

## Odd for train, even for eval
# vg_eval = vg_train[::2]; vg_train = vg_train[1::2]
# vds_eval = vds_train[::2]; vds_train = vds_train[1::2]
# id_eval = id_train[::2]; id_train = id_train[1::2]
# data_arrays_train = [vg_train, vds_train, id_train]
# data_arrays_eval = [vg_eval, vds_eval, id_eval]

## Check shape of train and eval dataset
print('--- Train/Eval shape: ')
print(
	data_arrays_train[0].shape, 
	data_arrays_train[1].shape, 
	data_arrays_train[2].shape
)
print(
	data_arrays_eval[0].shape,
	data_arrays_eval[1].shape,
	data_arrays_eval[2].shape
)

scale, vg_shift = preproc.compute_dc_meta(*data_arrays_train)
preproc_param = {
	'scale' : scale, 
	'vg_shift' : vg_shift, 
}
print(preproc_param)

## Saving the preproc param
preproc_data_arrays_train = preproc.dc_iv_preproc(
	data_arrays_train[0], data_arrays_train[1], data_arrays_train[2], 
	preproc_param['scale'], 
	preproc_param['vg_shift']
)
preproc_data_arrays_eval = preproc.dc_iv_preproc(
	data_arrays_eval[0], data_arrays_eval[1], data_arrays_eval[2],
	preproc_param['scale'], 
	preproc_param['vg_shift']
)

# Only expand the dim if the number of dimension is 1
preproc_data_arrays_train = [np.expand_dims(
	x, axis=1) if x.ndim == 1 else x for x in preproc_data_arrays_train]
preproc_data_arrays_eval = [np.expand_dims(
	x, axis=1) if x.ndim == 1 else x for x in preproc_data_arrays_eval]
# Write to database
if os.path.isfile(db_path+'train.minidb'):
	print("XXX Delete the old train database...")
	os.remove(db_path+'train.minidb')
if os.path.isfile(db_path+'eval.minidb'):
	print("XXX Delete the old eval database...")
	os.remove(db_path+'eval.minidb')
data_reader.write_db('minidb', db_path+'train.minidb', preproc_data_arrays_train)
data_reader.write_db('minidb', db_path+'eval.minidb', preproc_data_arrays_eval)
pickle.dump(preproc_param, open(db_path+'preproc_param.p', 'wb'))


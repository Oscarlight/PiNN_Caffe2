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

vds = np.linspace(0.0, 0.4, 41)
vbg = np.linspace(0.0, 0.4, 41)
vtg = np.linspace(0.0, 0.4, 67)
iter_lst = list(product(vds, vbg, vtg))
vds_train = np.array([e[0] for e in iter_lst], dtype=np.float32)
vbg_train = np.array([e[1] for e in iter_lst], dtype=np.float32)
vtg_train = np.array([e[2] for e in iter_lst], dtype=np.float32)
vss_train = np.zeros_like(vtg_train)
v_train = np.column_stack((vds_train, vbg_train, vtg_train, vss_train))

print(v_train.shape)

terminal_list = ['b', 'd', 'g', 's']
for term in terminal_list:
	cv_files = glob.glob('./transiXOR_data/*_C'+term+'?_*')
	capa_data = [np.expand_dims(
		np.load(f).flatten(), axis=1).astype(np.float32) for f in cv_files]
	c_train = np.column_stack(capa_data) # capacitances
	scale, vg_shift = preproc.compute_ac_meta(v_train, c_train)
	# print(c_train.shape)
	# print(scale, vg_shift)
	v_train, c_train = preproc.ac_qv_preproc(v_train, c_train, scale, vg_shift)
	# print(v_train.shape)
	# print(c_train.shape)
	s_train = np.ones((v_train.shape[0], 1)) # selector aka adjoint input
	## get eval and train
	v_eval = v_train[::2]; v_train = v_train[1::2]
	c_eval = c_train[::2]; c_train = c_train[1::2]
	s_eval = s_train[::2]; s_train = s_train[1::2]
	## Saving 
	data_reader.write_db(
		'minidb', term+'_train.minidb', 
		[v_train, s_train, c_train]
	)
	data_reader.write_db(
		'minidb', term+'_eval.minidb', 
		[v_eval, s_eval, c_eval]
	)
	preproc_param = {
		'scale' : scale, 
		'vg_shift' : vg_shift, 
	}
	pickle.dump(preproc_param, open(term + '_preproc_param.p', 'wb'))
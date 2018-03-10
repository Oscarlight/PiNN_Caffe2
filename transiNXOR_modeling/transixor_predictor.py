import sys
sys.path.append('../')
import numpy as np
from itertools import product
from pinn_api import predict_ids_grads, predict_ids
import matplotlib.pyplot as plt
import glob

## ------------  True data  ---------------
ids_file = glob.glob('./transiXOR_data/current_D9.npy')
# ids_file = glob.glob('./transiXOR_data/*_id_*.npy')
# vds, vbg, vtg, id
ids_data = np.load(ids_file[0])
print(ids_data.shape)

## ------------  Prediction ---------------
# vds = np.linspace(-0.1, 0.3, 41)
# vbg = np.linspace(0.1, 0.1, 1)
# vtg = np.linspace(0.2, 0.2, 1)
vds = np.linspace(0.2, 0.2, 1)
vbg = np.linspace(0.1, 0.1, 1)
vtg = np.linspace(-0.1, 0.3, 41)

iter_lst = list(product(vds, vbg, vtg))
vds_pred = np.expand_dims(np.array([e[0] for e in iter_lst], dtype=np.float32), axis=1)
vbg_pred = np.array([e[1] for e in iter_lst], dtype=np.float32)
vtg_pred = np.array([e[2] for e in iter_lst], dtype=np.float32)
vg_pred = np.column_stack((vtg_pred, vbg_pred))

vg_pred = np.sum(vg_pred, axis=1, keepdims=True)

# vg_pred = np.sum(vg_pred, axis=1, keepdims=True)

## If trained with adjoint builder
# ids_pred, _, _ = predict_ids_grads(
# 	'./transiXOR_Models/bise_h16', vg_pred, vds_pred)

## If trained with origin builder
ids_pred = predict_ids(
	'./transiXOR_Models/bise_ext_sym_h264_0', vg_pred, vds_pred)

# ids_true = ids_data[:, 30, 20]
# vds_true = np.linspace(-0.1, 0.3, 41)
# plt.plot(vds, ids_pred, 'r')
# plt.plot(vds_true, ids_true)
# plt.show()
# plt.semilogy(vds, np.abs(ids_pred), 'r') 
# plt.semilogy(vds_true, np.abs(ids_true))
# plt.show()

ids_true = ids_data[30, 20, :]
vtg_true = np.linspace(-0.1, 0.3, 41)
plt.plot(vtg, ids_pred, 'r')
plt.plot(vtg_true, ids_true)
plt.show()
plt.semilogy(vtg, np.abs(ids_pred), 'r') 
plt.semilogy(vtg_true, np.abs(ids_true))
plt.show()

## Point test

ids_pred = predict_ids(
	'./transiXOR_Models/bise_ext_sym_h264_0',
	np.array([0.2+0.2]), np.array([0.2]))
print(ids_pred)

import sys
sys.path.append('../')
import numpy as np
from itertools import product
from pinn_api import predict_ids_grads, predict_ids
import matplotlib.pyplot as plt
import glob

## ------------  True data  ---------------
ids_file = glob.glob('./transiXOR_data/current.npy')
# ids_file = glob.glob('./transiXOR_data/*_id_*.npy')
# vds, vbg, vtg, id
ids_data = np.load(ids_file[0])

## ------------  Prediction ---------------
vds = np.linspace(0.2, 0.2, 1)
vbg = np.linspace(0.0, 0.2, 21)
vtg = np.linspace(0.1, 0.1, 1)
iter_lst = list(product(vds, vbg, vtg))
vds_pred = np.expand_dims(np.array([e[0] for e in iter_lst], dtype=np.float32), axis=1)
vbg_pred = np.array([e[1] for e in iter_lst], dtype=np.float32)
vtg_pred = np.array([e[2] for e in iter_lst], dtype=np.float32)
vg_pred = np.column_stack((vtg_pred, vbg_pred))

# vg_pred = np.sum(vg_pred, axis=1, keepdims=True)

## If trained with adjoint builder
# ids_pred, _, _ = predict_ids_grads(
# 	'./transiXOR_Models/bise_h16', vg_pred, vds_pred)

## If trained with origin builder
ids_pred = predict_ids(
	'./transiXOR_Models/bise_h216_0', vg_pred, vds_pred)

# plt.plot(ids_pred, 'r')
# plt.plot(np.abs(ids_data[20, :, 10]))
plt.semilogy(np.abs(ids_pred), 'r')
plt.semilogy(np.abs(ids_data[20, :, 10]))
plt.show()

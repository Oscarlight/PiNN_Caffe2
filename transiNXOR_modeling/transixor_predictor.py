import sys
sys.path.append('../')
import numpy as np
from itertools import product
from pinn_api import predict_ids_grads

vds = np.linspace(0, 0.4, 41)
vbg = np.linspace(0, 0.4, 41)
vtg = np.linspace(0, 0.4, 67)
iter_lst = list(product(vds, vbg, vtg))
vds_train = np.expand_dims(np.array([e[0] for e in iter_lst], dtype=np.float32), axis=1)
vbg_train = np.array([e[1] for e in iter_lst], dtype=np.float32)
vtg_train = np.array([e[2] for e in iter_lst], dtype=np.float32)
vg_train = np.column_stack((vtg_train, vbg_train))

ids_pred, _, _ = predict_ids_grads(
	'./transiXOR_Models/model_output_0', vg_train, vds_train)
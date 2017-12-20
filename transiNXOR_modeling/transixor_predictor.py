import sys
sys.path.append('../')
import numpy as np
from itertools import product
from pinn_api import predict_ids_grads
import matplotlib.pyplot as plt
## ------------  True data  ---------------


## ------------  Prediction ---------------
# vds = np.linspace(0, 0.4, 41)
vds = np.linspace(0.4, 0.4, 1)
# vbg = np.linspace(0, 0.4, 41)
vbg = np.linspace(0.0, 0.0, 1)
vtg = np.linspace(0, 0.4, 67)
iter_lst = list(product(vds, vbg, vtg))
vds_pred = np.expand_dims(np.array([e[0] for e in iter_lst], dtype=np.float32), axis=1)
vbg_pred = np.array([e[1] for e in iter_lst], dtype=np.float32)
vtg_pred = np.array([e[2] for e in iter_lst], dtype=np.float32)
vg_pred = np.column_stack((vtg_pred, vbg_pred))

ids_pred, _, _ = predict_ids_grads(
	'./transiXOR_Models/model_output_0', vg_pred, vds_pred)

plt.semilogy(ids_pred)
plt.show()

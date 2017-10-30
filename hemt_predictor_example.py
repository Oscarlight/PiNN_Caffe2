from pinn_api import predict_ids_grads, plot_iv
from pinn.exporter import load_init_net, read_param
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
#
load_init_net('model_output/GaN_HEMT_1_init')
plt.hist(
	read_param('adjoint/inter_embed_layer_0/w').flatten(), 
	bins=np.linspace(-4, 4, 50))
# plt.show()
# quit()
# ------------------------------------------------------
iter_lst = list(product(
	np.linspace(-1.0, 2.5, 50),
	np.linspace(0, 6, 50)
))
vg_pred = np.array([e[0] for e in iter_lst])
vd_pred = np.array([e[1] for e in iter_lst])
# vg_pred, vd_pred = vg, vd
ids, sig_grad, tanh_grad = predict_ids_grads(
	'model_output/GaN_HEMT_1', vg_pred, vd_pred)
print(min(tanh_grad))
# styles = ['vg_major_linear', 'vd_major_linear']
# plot_iv(vg_pred, vd_pred, ids, 
# 	save_name='model_output/GaN_HEMT_pred_id_',
# 	styles=styles)
# plot_iv(vg_pred, vd_pred, sig_grad, 
# 	save_name='model_output/GaN_HEMT_pred_idvg_',
# 	styles=styles,
# 	yLabel='$dI_{D}/dV_{g}$')
# plot_iv(vg_pred, vd_pred, tanh_grad, 
# 	save_name='model_output/GaN_HEMT_pred_idvd_',
# 	styles=styles,
# 	yLabel='$dI_{D}/dV_{d}$')
from pinn_api import predict_ids_grads, plot_iv
import numpy as np
from itertools import product

iter_lst = list(product(
	np.linspace(-1.0, 2.5, 50),
	np.linspace(0, 6, 50)
))
vg_pred = np.array([e[0] for e in iter_lst])
vd_pred = np.array([e[1] for e in iter_lst])
# vg_pred, vd_pred = vg, vd
ids, sig_grad, tanh_grad = predict_ids_grads('model_output/GaN_HEMT', vg_pred, vd_pred)
styles = ['vg_major_linear', 'vd_major_linear']
plot_iv(vg_pred, vd_pred, ids, 
	save_name='model_output/GaN_HEMT_pred_id_',
	styles=styles)
plot_iv(vg_pred, vd_pred, sig_grad, 
	save_name='model_output/GaN_HEMT_pred_idvg_',
	styles=styles,
	yLabel='$dI_{D}/dV_{g}$')
plot_iv(vg_pred, vd_pred, tanh_grad, 
	save_name='model_output/GaN_HEMT_pred_idvd_',
	styles=styles,
	yLabel='$dI_{D}/dV_{d}$')
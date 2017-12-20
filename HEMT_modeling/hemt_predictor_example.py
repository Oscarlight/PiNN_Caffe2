import sys
sys.path.append('../')
from pinn_api import predict_ids_grads, plot_iv
from pinn.exporter import load_init_net, read_param
from pinn import parser
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
CHANNEL_LENGTH = 0.07
## ------------ Check the weight ------------
#
# load_init_net('model_output_0/HEMT_DC_5')
# plt.hist(
# 	read_param('adjoint/inter_embed_layer_0/w').flatten(), 
# 	bins=np.linspace(-4, 4, 50))
# plt.show()
# quit()
IVPLOT = True
## ------------ The train and eval data ------------
if IVPLOT:
	# data_arrays format: [vg, vd, ids]
	data_arrays = parser.read_dc_iv_mdm('./HEMT_bo/Id_vs_Vd_at_Vg.mdm')
	vg, vd, ids = data_arrays[0], data_arrays[1], data_arrays[2]
	np.random.seed = 42  # use same seed as during training
	permu = np.random.permutation(len(data_arrays[0]))
	data_arrays = [e[permu] for e in data_arrays]
	data_arrays_eval = np.array([e[0:100] for e in data_arrays])
	data_arrays_train = np.array([e[100:] for e in data_arrays])


	## ------------ Original data -----------------------
	# plot_iv(
	# 	vg, vd, ids/CHANNEL_LENGTH,
	# 	save_name='GaN_HEMT_origin_id_',
	# 	yLabel='',
	# )
	# quit()

	## ------------ Predication data ----------------
	# Extension
	USE_EXTENSION = False
	if USE_EXTENSION:
		vg_unique = np.unique(vg)
		vd_unique = np.unique(vd)
		vg_range = np.max(vg_unique) - np.min(vg_unique)
		vd_range = np.max(vd_unique) - np.min(vd_unique)
		extension_vg = vg_range * 0.2; dvg = vg_unique[1] - vg_unique[0]
		extension_vd = vd_range * 0.2; dvd = vd_unique[1] - vd_unique[0]
		print(extension_vg); print(dvg, vg_unique.shape);
		print(extension_vd); print(dvd, vd_unique.shape); 
		extended_vg_unique = np.linspace(
			vg_unique[0] - 2 * extension_vg, # - %40
			vg_unique[-1] + 2 * extension_vg,
			2*(vg_unique.shape[0] + 8) - 1 )
		extended_vd_unique = np.linspace(
			vd_unique[0],
			vd_unique[-1] + 4 * extension_vd,
			vd_unique.shape[0] + 40)

		iter_lst = list(product(
			extended_vg_unique, 
			extended_vd_unique,
			)
		)
		vg_pred = np.array([e[0] for e in iter_lst])
		vd_pred = np.array([e[1] for e in iter_lst])

	else:
		# Use the origin data
		vg_pred = vg
		vd_pred = vd
	# quit()
	ids_pred, _, _ = predict_ids_grads(
		'./HEMT_Models/model_output_1/HEMT_DC_5', vg_pred, vd_pred)

	# styles = ['vd_major_linear', 'vd_major_log']
	# styles = ['vg_major_log', 'vg_major_linear']

	plot_iv(
		np.array(vg_pred), np.array(vd_pred), ids_pred/CHANNEL_LENGTH,
		vg_comp = [data_arrays_eval[0], data_arrays_train[0]], 
		vd_comp = [data_arrays_eval[1], data_arrays_train[1]], 
		ids_comp = [data_arrays_eval[2]/CHANNEL_LENGTH, data_arrays_train[2]/CHANNEL_LENGTH],
		save_name='GaN_HEMT_pred_id_',
		yLabel='',
		# styles=styles
	)

	quit()


GRAD1 = False
if GRAD1:
	## --------------- Plot the gradients ------------------
	# self defined input
	iter_lst = list(product(
		np.linspace(-1.0, 2.5, 100),
		# np.linspace(3, 4, 2)
		np.linspace(0, 6, 21)
	))
	vg_pred = np.array([e[0] for e in iter_lst])
	vd_pred = np.array([e[1] for e in iter_lst])
	ids_pred, sig_grad, _ = predict_ids_grads(
		'./HEMT_Models/HEMT_DC_5', vg_pred, vd_pred)

	# gm from Id-Vg 
	data_arrays = parser.read_dc_iv_mdm('./HEMT_bo/Id_vs_Vg_at_Vd.mdm')
	vg, vd, ids = data_arrays[0], data_arrays[1], data_arrays[2]
	# print(ids/CHANNEL_LENGTH)
	plot_iv(
		vg, vd, ids/CHANNEL_LENGTH,
		save_name='GaN_HEMT_origin_id_',
		yLabel='',
		styles=['vd_major_linear']
	)
	gm = np.diff(ids)/np.diff(vg)
	styles = ['vd_major_linear']
	plot_iv(
		vg_pred, vd_pred, sig_grad/CHANNEL_LENGTH, 
		vg_comp=[vg[1:-1]], vd_comp=[vd[1:-1]], ids_comp=[gm[:-1]/CHANNEL_LENGTH],
		save_name='GaN_HEMT_pred_idvg_',
		styles=styles,
		yLabel='')
	quit()
	# self defined input
	iter_lst = list(product(
		np.linspace(-1.0, 2.5, 21),
		# np.linspace(2.4, 2.5, 2),
		np.linspace(0, 6, 100)
	))
	vg_pred = np.array([e[0] for e in iter_lst])
	vd_pred = np.array([e[1] for e in iter_lst])
	ids_pred, _, tanh_grad = predict_ids_grads(
		'./HEMT_Models/model_output_1/HEMT_DC_5', vg_pred, vd_pred)

	styles = ['vg_major_linear']
	plot_iv(
		vg_pred, vd_pred, tanh_grad/CHANNEL_LENGTH, 
		save_name='GaN_HEMT_pred_idvd_',
		styles=styles,
		yLabel='')


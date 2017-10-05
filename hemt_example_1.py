from dc_iv_api import DCModel, plot_iv, predict_ids
import pinn.parser as parser
import pinn.preproc as preproc
import pinn.exporter as exporter
import numpy as np
import time

TRAIN = True
TEST_ONLY = not TRAIN

# ----------------- Preprocessing --------------------
# data_arrays format: [vg, vd, ids]
data_arrays = parser.read_dc_iv_csv('./HEMT_bo/DC_IV.csv')
data_arrays = preproc.truncate(data_arrays, (-2, -1.2), 0)
vg, vd, ids = data_arrays[0], data_arrays[1], data_arrays[2]
# plot_iv(vd, vg, ids)
scale, vg_shift = preproc.compute_dc_meta(*data_arrays)
preproc_param = {
	'scale' : scale, 
	'vg_shift' : vg_shift, 
	'max_loss_scale' : 1e4,
}
permu = np.random.permutation(len(data_arrays[0]))
data_arrays = [e[permu] for e in data_arrays]

# ----------------- Train + Eval ---------------------
if TRAIN:
	dc_model = DCModel('HEMT_DC_1_L1_Weighted')
	dc_model.add_data('train', data_arrays, preproc_param)
	# plot_iv(*dc_model.preproc_data_arrays)
	# quit()
	dc_model.build_nets(
		hidden_sig_dims=[16,1],  # Need to be fine-tuned
		hidden_tanh_dims=[16,1],
		batch_size=732,
		weight_optim_method = 'AdaGrad',
		weight_optim_param = {'alpha':0.2, 'epsilon':1e-4},
		bias_optim_method = 'AdaGrad',
		bias_optim_param = {'alpha':0.2, 'epsilon':1e-4} 
	)
	start = time.time()
	dc_model.train_with_eval(
		num_epoch=int(1e6),
		report_interval=100,
	)
	end = time.time()
	print('Elapsed time: ' + str(end - start))
	# ----------------- Inspection ---------------------
	# dc_model.draw_nets()
	dc_model.plot_loss_trend()

	# # ----------------- Deployment ---------------------
	_, pred_ids = dc_model.predict_ids(vg, vd)
	plot_iv(
		vg, vd, ids,
		vg_comp=vg, 
		vd_comp=vd,
		ids_comp=pred_ids,
		save_name='HEMT_DC_1_L1_Weighted'
	)

# -------------- Load Saved Model ------------------
if TEST_ONLY:
	from itertools import product
	iter_lst = list(product(
		np.linspace(-1.2, 0, 100),
		np.linspace(0, 6, 100)
	))
	vg_pred = np.array([e[0] for e in iter_lst])
	vd_pred = np.array([e[1] for e in iter_lst])
	vg_pred, vd_pred = vg, vd
	_, pred_ids = predict_ids('HEMT_DC_1_L1_Weighted', vg_pred, vd_pred)
	plot_iv(vg_pred, vd_pred, pred_ids,
		vg_comp=vg, 
		vd_comp=vd,
		ids_comp=ids)
	
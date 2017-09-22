from dc_iv_api import DCModel, plot_iv, predict_ids
import pinn.parser as parser
import pinn.preproc as preproc
import pinn.exporter as exporter
import numpy as np

# TODO: 
# Train on
# './HEMT_bo/Id_vs_Vd_at_Vg.mdm'
# Test on
# './HEMT_bo/Id_vs_Vg_at_Vd.mdm'

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
	'preproc_slope' : 10, 
	'preproc_threshold' : 0.8
}
permu = np.random.permutation(len(data_arrays[0]))
data_arrays = [e[permu] for e in data_arrays]
# ----------------- Train + Eval ---------------------
dc_model = DCModel('HEMT_DC_1')
dc_model.add_data('train', data_arrays, preproc_param)
# plot_iv(*dc_model.preproc_data_arrays, styles=['vd_major_log'])
dc_model.build_nets(
	hidden_sig_dims=[7, 7, 1],  # Need to be fine-tuned
	hidden_tanh_dims=[7, 7, 1],
	batch_size=256,
	weight_optim_method = 'AdaGrad',
	weight_optim_param = {'alpha':0.005, 'epsilon':1e-4},
	bias_optim_method = 'AdaGrad',
	bias_optim_param = {'alpha':0.05, 'epsilon':1e-4} 
)

dc_model.train_with_eval(
	num_epoch=int(1e7),  # several hrs training time
	report_interval=0,
)

# ----------------- Inspection ---------------------
dc_model.draw_nets()
# dc_model.plot_loss_trend()

# ----------------- Deployment ---------------------
_, pred_ids = dc_model.predict_ids(vg, vd)
plot_iv(
	vg, vd, ids,
	vg_comp=vg, 
	vd_comp=vd,
	ids_comp=pred_ids,
)

# -------------- Load Saved Model ------------------
vg_pred = np.linspace(-1.2, 0, 1000)
vd_pred = np.array([0.6]*1000)
_, pred_ids = predict_ids('HEMT_DC_1', vg_pred, vd_pred)
plot_iv(vg, vd, pred_ids, styles=['vg_major_log'])


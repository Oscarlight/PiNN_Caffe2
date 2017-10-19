from dc_iv_api import DCModel, plot_iv, predict_ids
import pinn.parser as parser
import pinn.preproc as preproc
import pinn.exporter as exporter
import numpy as np

# ----------------- Preprocessing --------------------
# data_arrays format: [vg, vd, ids]
data_arrays = parser.read_dc_iv_mdm('./HEMT_bo/Id_vs_Vd_at_Vg.mdm')
data_arrays_test = parser.read_dc_iv_mdm('./HEMT_bo/Id_vs_Vg_at_Vd.mdm')
#data_arrays = preproc.truncate(data_arrays, (-2, -1.2), 0)
vg, vd, ids = data_arrays[0], data_arrays[1], data_arrays[2]
vgtest, vdtest, idstest = data_arrays_test[0], data_arrays_test[1], data_arrays_test[2]
# plot_iv(vg, vd, ids)
# quit()
scale, vg_shift = preproc.compute_dc_meta(*data_arrays)
preproc_param = {
	'scale' : scale, 
	'vg_shift' : vg_shift, 
}
np.random.seed = 42
permu = np.random.permutation(len(data_arrays[0]))
data_arrays = [e[permu] for e in data_arrays]
data_arrays_eval = [e[0:100] for e in data_arrays]
data_arrays_train = [e[100:] for e in data_arrays]

# ----------------- Train + Eval ---------------------
dc_model = DCModel('HEMT_DC_2_L1_Weighted')
dc_model.add_data('train', data_arrays_train, preproc_param)
dc_model.add_data('eval',data_arrays_eval, preproc_param)
# plot_iv(*dc_model.preproc_data_arrays)

dc_model.build_nets(
	hidden_sig_dims=[16, 1],
	hidden_tanh_dims=[16, 1],
	train_batch_size=461,
	eval_batch_size=100,
	weight_optim_method = 'AdaGrad',
	weight_optim_param = {'alpha':0.1, 'epsilon':1e-4},
	bias_optim_method = 'AdaGrad',
	bias_optim_param = {'alpha':0.1, 'epsilon':1e-4},
	loss_function = 'scaled_l1', # or 'scaled_l2'
	max_loss_scale = 5e5, 
	# to improve subthreshold modeling accuracy
	# for 'scaled_l1': max_loss_scale: ~5e5	
	# for 'scaled_l2' : max_loss_scale: > 5e9
	# Note 'scaled_l1' performed better than 'scaled_l2' 
)

dc_model.train_with_eval(
	num_epoch=int(1e6),
	report_interval=1e3,
	eval_during_training=True
)

# # ----------------- Inspection ---------------------
# dc_model.draw_nets()
dc_model.plot_loss_trend()

# # ----------------- Deployment ---------------------
_, pred_ids = dc_model.predict_ids(vg, vd)
plot_iv(
	vg, vd, ids,
	vg_comp=vg, 
	vd_comp=vd, 
	ids_comp=pred_ids,
)

#exporter.load_net(dc_model.model_name+'_init', dc_model.model_name+'_predict')


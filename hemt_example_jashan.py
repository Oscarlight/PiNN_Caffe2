from dc_iv_api import DCModel, plot_iv, predict_id_test
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
data_arrays = parser.read_dc_iv_mdm('./HEMT_bo/Id_vs_Vd_at_Vg.mdm')
data_arrays_test = parser.read_dc_iv_mdm('./HEMT_bo/Id_vs_Vg_at_Vd.mdm')
#data_arrays = preproc.truncate(data_arrays, (-2, -1.2), 0)
vg, vd, ids = data_arrays[0], data_arrays[1], data_arrays[2]
# plot_iv(vg, vd, ids)
scale, vg_shift = preproc.compute_dc_meta(*data_arrays)
preproc_param = {
	'scale' : scale, 
	'vg_shift' : vg_shift, 
	'preproc_slope' : 0, 
	'preproc_threshold' : 0.5
}
permu = np.random.permutation(len(data_arrays[0]))
data_arrays = [e[permu] for e in data_arrays]
# # ----------------- Train + Eval ---------------------
dc_model = DCModel('HEMT_DC_1')
dc_model.add_data('train', data_arrays, preproc_param)
dc_model.add_data('eval', data_arrays_test)
dc_model.build_nets(
	hidden_sig_dims=[3, 1],
	hidden_tanh_dims=[2, 1],
	batch_size=256,
	weight_optim_method = 'AdaGrad',
	weight_optim_param = {'alpha':0.01, 'epsilon':1e-4},
	bias_optim_method = 'AdaGrad',
	bias_optim_param = {'alpha':0.1, 'epsilon':1e-4} 
)

dc_model.train_with_eval(
	num_epoch=500,
	report_interval=10,
)

# # ----------------- Inspection ---------------------
dc_model.draw_nets()
dc_model.plot_loss_trend()

# # ----------------- Deployment ---------------------
intern_ids, pred_ids = dc_model.predict_ids(vg, vd)
# plot_iv(
# 	vd, vg, ids,
# 	vg_comp=vd, 
# 	vd_comp=vg, 
# 	ids_comp=pred_ids,
# )

#exporter.load_net(dc_model.model_name+'_init', dc_model.model_name+'_predict')


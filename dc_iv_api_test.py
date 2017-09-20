from dc_iv_api import DCModel, plot_iv, predict_id_test
import parser
import preproc
import numpy as np
import exporter
# TODO: Test on
# './HEMT_bo/Id_vs_Vd_at_Vg.mdm'
# './HEMT_bo/Id_vs_Vg_at_Vd.mdm'

# ----------------- Preprocessing --------------------
data_arrays = parser.read_dc_iv_csv('./HEMT_bo/DC_IV.csv')
scale, vg_shift = preproc.compute_dc_meta(*data_arrays)
preproc_param = {
	'scale' : scale, 
	'vg_shift' : vg_shift, 
	'preproc_slope' : 5, 
	'preproc_threshold' : 0.1
}
# ----------------- Train + Eval ---------------------
dc_model = DCModel('hemt_dc_test_2')
dc_model.add_data('train', data_arrays, preproc_param)
dc_model.build_nets(
	hidden_sig_dims=[4, 4, 1],
	hidden_tanh_dims=[3, 3, 1],
	batch_size=64,
	weight_optim_method = 'AdaGrad',
	weight_optim_param = {'alpha':0.005, 'epsilon':1e-4},
	bias_optim_method = 'AdaGrad',
	bias_optim_param = {'alpha':0.05, 'epsilon':1e-4} 
)
#INIT_NET = '/Users/jashansinghal/Box Sync/PINNGit/PiNN_Caffe2/init_net.pb'
#PREDICT_NET = '/Users/jashansinghal/Box Sync/PINNGit/PiNN_Caffe2/predict_net.pb'
dc_model.train_with_eval(
	num_epoch=10,
	report_interval=0,
)

# ----------------- Inspection ---------------------
# dc_model.draw_nets()
# dc_model.plot_loss_trend()

# ----------------- Deployment ---------------------
vg = data_arrays[0]
vd = data_arrays[1]
ids = data_arrays[2]
intern_ids, pred_ids = dc_model.predict_id(vg, vd)
# plot_iv(
# 	vd, vg, ids,
# 	vg_comp=vd, 
# 	vd_comp=vg, 
# 	ids_comp=pred_ids,
# )
# ----------------- Test ---------------------
vg = data_arrays[0]
vd = data_arrays[1]
ids = data_arrays[2]
intern_ids, pred_ids = predict_id_test(dc_model, vg, vd)
plot_iv(
	vd, vg, ids,
	vg_comp=vd, 
	vd_comp=vg, 
	ids_comp=pred_ids,
)


#exporter.load_net(dc_model.model_name+'_init', dc_model.model_name+'_predict')


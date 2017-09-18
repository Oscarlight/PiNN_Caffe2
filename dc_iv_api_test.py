from dc_iv_api import DCModel, plot_iv
import numpy as np
# TODO: Test on
# './HEMT_bo/Id_vs_Vd_at_Vg.mdm'
# './HEMT_bo/Id_vs_Vg_at_Vd.mdm'

# ----------------- Train + Eval time ---------------------
dc_model = DCModel(
	'hemt_dc_test_2', 
	train_file_name='./HEMT_bo/DC_IV.csv',
)

dc_model.build_nets(
	hidden_sig_dims=[3, 1],
	hidden_tanh_dims=[3, 1],
	batch_size=10,
	optim_param = {'alpha':0.01, 'epsilon':1e-4} 
)
# If you want to take a look at the nets
# dc_model.draw_nets()

dc_model.train_with_eval(
	num_epoch=1000, 
	report_interval=0,
)
# dc_model.plot_loss_trend()

# ----------------- Deployment time ---------------------
dataset = np.genfromtxt('./HEMT_bo/DC_IV.csv', delimiter=",", dtype=None)
vg = []
vd = []
ids = []
for i in range(1, len(dataset)):
	vg.append(float(dataset[i][0]))
	vd.append(float(dataset[i][1]))
	ids.append(float(dataset[i][2]))
vg = np.array(vg)
vd = np.array(vd)
ids = np.array(ids)
pred_ids = dc_model.predict_id(vg, vd)
plot_iv(
	vg, vd, ids, 
	vg_comp=vg, 
	vd_comp=vd, 
	ids_comp=pred_ids
)

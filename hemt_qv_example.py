from ac_qv_api import ACQVModel, plot_iv, predict_ids
import pinn.parser as parser
import pinn.preproc as preproc
import pinn.exporter as exporter
import pinn.deembed as deembed
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

data_arrays = deembed.deembed(parser.read_s_par_mdm, './HEMT_bo/s_at_f_vs_VgVd.mdm', 1e-12, 1e-12, 1, 2)

#vg = np.array([[vg_value] for vg_value in vg], dtype = np.float32)
#vd = np.array([[vd_value] for vd_value in vd], dtype = np.float32)
#x_array = np.linspace(-0.8, 0.8, 100)
#origin_input = np.array([[x] for x in x_array], dtype = np.float32)
#adjoint_label = np.array(
#	[[(1 - np.exp(10*x)) / (1 + np.exp(10*x))] for x in x_array], 
#	dtype = np.float32
#)
#adjoint_input = np.ones((100,1), dtype = np.float32)
voltage = np.concatenate(
	(np.expand_dims(data_arrays[0], axis=1),
	 np.expand_dims(data_arrays[1], axis=1)), 
	axis=1
)
capas = np.array(data_arrays[6])

scale, vg_shift = preproc.compute_ac_meta(data_arrays[0], data_arrays[1], capas)

preproc_param = {
	'scale': scale,
	'vg_shift': vg_shift,
}
ac_model = ACQVModel('ac_model', input_dim=2, output_dim=1)
ac_model.add_data('train', [voltage, capas], preproc_param)
ac_model.build_nets([10, 10, 10], batch_size = 1275)
ac_model.train_with_eval(num_epoch = 10000, report_interval = 0)
ac_model.draw_nets() 
ac_model.plot_loss_trend()

pred_qs, ori_qs = ac_model.predict_qs(data_arrays[0], data_arrays[1])


plot_iv(
	data_arrays[0], data_arrays[1], data_arrays[2],
	vg_comp = data_arrays[0],
	vd_comp = data_arrays[1],
	ids_comp = ori_qs
)



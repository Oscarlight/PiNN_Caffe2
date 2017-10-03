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
data_arrays = [data_arrays[0]] + [data_arrays[1]] + [data_arrays[5]]

scale, vg_shift = preproc.compute_ac_meta(data_arrays[0], data_arrays[1], data_arrays[2], checkaccuracy = True)
preproc_param = {
	'scale': scale,
	'vg_shift': vg_shift,
	'preproc_slope': 0,
	'preproc_threshold': 0
}
data_arrays[2] = np.asarray(data_arrays[2])[:,0]
ac_model = ACQVModel('ac_model')
ac_model.add_data('train', data_arrays, preproc_param)
ac_model.build_nets([5, 5], batch_size = 10)
ac_model.train_with_eval(num_epoch = 10, report_interval = 0)
ac_model.draw_nets() 

pred_qs, ori_gradient = ac_model.predict_qs(data_arrays[0], data_arrays[1])
plot_iv(
	data_arrays[0], data_arrays[1], data_arrays[2],
	vg_comp = data_arrays[0],
	vd_comp = data_arrays[1],
	ids_comp = pred_qs
)

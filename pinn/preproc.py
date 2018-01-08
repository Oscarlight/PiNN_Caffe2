import numpy as np
import matplotlib.pyplot as plt

# slope and thre are deprecated
def dc_iv_preproc(
	vg, vd, ids, 
	scale, shift, 
	**kwarg
):
	if len(kwarg) > 0:
		print('[WARNING]: slope and threshold preprocessing is no longer supported!')
	# print(scale['vg'])
	preproc_vg = (vg-shift) / scale['vg']
	preproc_vd = vd / scale['vd']
	preproc_id = ids / scale['id']
	return preproc_vg, preproc_vd, preproc_id

def get_restore_id_func(scale, *arg, **kwarg):
	if (len(arg) > 0 or len(kwarg) > 0):
		print('[WARNING]: slope and threshold preprocessing is no longer supported!')
	def restore_id_func(ids, *arg):
		return ids * scale['id']
	def get_restore_id_grad_func(sig_grad, tanh_grad):
		return (sig_grad * scale['id'] / scale['vg'],
			tanh_grad * scale['id'] / scale['vd'])

	return restore_id_func, get_restore_id_grad_func



def compute_dc_meta(vg, vd, ids):
	vg_shift = np.median(vg,axis=0)

	vg_scale = np.maximum(
		abs(np.max(vg,axis=0)-vg_shift), 
		abs(np.min(vg,axis=0)-vg_shift)
	)
	vd_scale = np.maximum(
		abs(np.max(vd,axis=0)), 
		abs(np.min(vd,axis=0))
	)
	id_scale = np.maximum(
		abs(np.max(ids,axis=0))/0.75, 
		abs(np.min(ids,axis=0))/0.75
	)
	## replace 0 to 1
	vg_scale[vg_scale == 0.0] = 1.0
	vd_scale[vd_scale == 0.0] = 1.0
	id_scale[id_scale == 0.0] = 1.0
	scale = {'vg':vg_scale, 'vd':vd_scale, 'id':id_scale}

	return scale, vg_shift


def truncate(data_arrays, truncate_range, axis):
	# The data within the truncate_range will be removed, the rest will be returned.

	assert (truncate_range[0] <= np.max(data_arrays[axis]) and truncate_range[1] > np.min(data_arrays[axis])),\
		'Truncate range is out of data range, abort!'

	index = np.logical_not(
		np.logical_and(
			data_arrays[axis] > truncate_range[0],
			data_arrays[axis] <= truncate_range[1]
		)
	)

	return [e[index] for e in data_arrays]

#AC QV preproc
def ac_qv_preproc(voltages, grads, scale, shift):
	print('[WARNING:ac_qv_preproc]Precondition: the first column of voltage is Vd, and last column of voltages is Vs')
	voltages[:,1:-1] = (voltages[:,1:-1]-shift) / scale['vg']
	voltages[:,0:1] /= scale['vd']
	preproc_grads = grads/scale['grads']
	return voltages, preproc_grads

def get_restore_q_func(
		scale, shift
):
	def restore_integral_func(integrals):
		ori_integral = integrals*scale['grads']*scale['vd']*scale['vg']
		return ori_integral
	def restore_gradient_func(gradient):
		ori_gradient =  gradient * scale['grads']
		return ori_gradient
	return restore_integral_func, restore_gradient_func
	
def compute_ac_meta(voltages, grads):
	print('[WARNING:compute_ac_meta]Precondition: the first column of voltage is Vd, and last column of voltages is Vs')
	vd = voltages[:, 0:1]
	vg = voltages[:, 1:-1]
	
	vg_shift = np.median(vg,axis=0)
	vg_scale = np.maximum(
		abs(np.max(vg,axis=0)-vg_shift), 
		abs(np.min(vg,axis=0)-vg_shift)
	)
	vd_scale = np.maximum(
		abs(np.max(vd,axis=0)), 
		abs(np.min(vd,axis=0))
	)
	# print(vd_scale)
	grads_scale = np.maximum(
		abs(np.max(grads,axis=0))/0.85, 
		abs(np.min(grads,axis=0))/0.85
	)
	## replace 0 to 1
	vg_scale[vg_scale == 0.0] = 1.0
	vd_scale[vd_scale == 0.0] = 1.0
	grads_scale[grads_scale == 0.0] = 1.0
 
	scale = {'vg':vg_scale, 'vd':vd_scale, 'grads':grads_scale}
	return scale, vg_shift



  


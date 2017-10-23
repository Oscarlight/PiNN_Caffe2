import numpy as np
import matplotlib.pyplot as plt

# slope and thre are deprecated
def dc_iv_preproc(
	vg, vd, ids, 
	scale, shift, 
	**kwarg
):
	if len(kwarg) > 0:
		print('[WARNING]: slope and threshold preprocessing is deprecated!')
	preproc_vg = (vg-shift) / scale['vg']
	preproc_vd = vd / scale['vd']
	preproc_id = ids / scale['id']
	return preproc_vg, preproc_vd, preproc_id

def get_restore_id_func(scale, *arg, **kwarg):
	if (len(arg) > 0 or len(kwarg) > 0):
		print('[WARNING]: slope and threshold preprocessing is deprecated!')
	def restore_id_func(ids, *arg):
		return ids * scale['id']
	def get_restore_id_grad_func(sig_grad, tanh_grad):
		return (sig_grad * scale['id'] / scale['vg'],
			tanh_grad * scale['id'] / scale['vd'])

	return restore_id_func, get_restore_id_grad_func



def compute_dc_meta(vg, vd, ids):
	vg_shift = np.median(vg)-0.0
	vg_scale = max(abs(np.max(vg)-vg_shift)/1.0, abs(np.min(vg)-vg_shift)/1.0)
	vd_scale = max(abs(np.max(vd))/1.0, abs(np.min(vd))/1.0)
	id_scale = max(abs(np.max(ids))/0.75, abs(np.min(ids))/0.75)

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
def ac_qv_preproc(preproc_voltages, gradient, scale, shift):
    preproc_voltages[:,0] = (preproc_voltages[:,0]-shift) / scale['vg']
    preproc_voltages[:,1] /= scale['vd']
    preproc_gradient = gradient/scale['q']
    return preproc_voltages, preproc_gradient

def restore_voltages(scale, shift, voltages):
	voltages[:, 0] = (voltages[:,0]*scale['vg']) + shift
	voltages[:, 1] *= scale['vd']
	return voltages

def get_restore_q_func(
        scale, shift
):
    def restore_integral_func(integrals):
        ori_integral = integrals*scale['q']*scale['vd']*scale['vg']
        return ori_integral
    def restore_gradient_func(gradient):
        ori_gradient =  gradient * scale['q']
        return ori_gradient
    return restore_integral_func, restore_gradient_func
    
def compute_ac_meta(voltage, gradient):
    vg = voltage[:, 0]
    vd = voltage[:, 1]
    vg_shift = np.median(vg)-0.0
    vg_scale = max(abs(np.max(vg)-vg_shift)/1.0, abs(np.min(vg)-vg_shift)/1.0)
    vd_scale = max(abs(np.max(vd))/1.0, abs(np.min(vd))/1.0)
    q_scale = max(abs(np.max(gradient))/0.75, abs(np.min(gradient))/0.75)   
    scale = {'vg':vg_scale, 'vd':vd_scale, 'q':q_scale}
    return scale, vg_shift


def plotgradient(vg, vd, gradient):
    #gradient = np.asarray(gradient)
    dqdvg = gradient[:, 0]
    dqdvd = gradient[:, 1]
    plt.plot(vg, dqdvg, 'r')
    plt.plot(vd, dqdvd, 'b')
    plt.show()

  


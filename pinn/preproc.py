import numpy as np

def dc_iv_preproc(
	vg, vd, ids, 
	scale, shift, 
	slope=0, threshold=0
):
    '''
    inputs:
        1) two numpy array features and labels
        2) arguments for pre-processing
    outputs:
        1) pre-process features and labels
        2) a function for restoring to the original data
    '''
    preproc_vg = (vg-shift) / scale['vg']
    preproc_vd = vd / scale['vd']
    preproc_id = ids / scale['id'] * (
    	np.power(10, -slope * (preproc_vg + threshold)) + 1
    	) if slope > 0 else ids/scale['id']

    return preproc_vg, preproc_vd, preproc_id

def get_restore_id_func(	
	scale, shift, 
	slope=0, threshold=0
):
	def restore_id_func(ids, vgs):
		# ori_vg = vgs * scale['vg'] + shift
		# ori_vd = vds * scale['vd']
		ori_id = ids * scale['id'] / (
			np.power(10, -slope*(vgs + threshold)) + 1
		) if slope > 0 else ids * scale['id']
		# return ori_vg, ori_vd, ori_id
		return ori_id
	return restore_id_func

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
def ac_qv_preproc(vg, vd, gradient, scale, shift, slope = 0, threshold = 0):
    preproc_vg = (vg-shift) / scale['vg']
    preproc_vd = vd / scale['vd']
    preproc_gradient = gradient / scale['q'] * (
        np.power(10, -slope * (preproc_vg + threshold)) + 1
        ) if slope > 0 else gradient/scale['q']

    return preproc_vg, preproc_vd, preproc_gradient

def get_restore_q_func(
        scale, shift,
        slope=0, threshold=0
):
        def restore_q_func(gradient, vgs):
                # ori_vg = vgs * scale['vg'] + shift
                # ori_vd = vds * scale['vd']
                ori_gradient = gradient * scale['q'] / (
                        np.power(10, -slope*(vgs + threshold)) + 1
                ) if slope > 0 else gradient * scale['q']
                # return ori_vg, ori_vd, ori_id
                return ori_gradient
        return restore_q_func
	
def compute_ac_meta(vg, vd, gradient):
    vg_shift = np.median(vg)-0.0
    vg_scale = max(abs(np.max(vg)-vg_shift)/1.0, abs(np.min(vg)-vg_shift)/1.0)
    vd_scale = max(abs(np.max(vd))/1.0, abs(np.min(vd))/1.0)
    integral = integrate(vg, vd, gradient)
    q_scale = max(abs(np.max(integral))/0.75, abs(np.min(integral))/0.75)	
    scale = {'vg':vg_scale, 'vd':vd_scale, 'q':q_scale}
    return scale, vg_shift

def integrate(vg, vd, gradient):
    qs = np.zeros(len(vg))
    for i in range(1, len(qs)):
        qs[i] = qs[i-1]+(vg[i]*gradient[i][0]+vd[i]*gradient[i][1]) 
    return qs

   

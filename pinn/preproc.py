import numpy as np

def dc_iv_preproc(
	vg, vd, ids, 
	scale, shift, 
    slope_vg=0, thre_vg=0,
    slope_vd=0, thre_vd=0,
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
    preproc_id = (ids / scale['id'] * 
    	(np.power(10, -slope_vg*(vg + thre_vg)) + 1) * 
        (np.power(10, -slope_vd*(abs(vd) - thre_vd)) + 1)
    	) if slope_vg > 0 or slope_vd > 0 else ids/scale['id']

    return preproc_vg, preproc_vd, preproc_id

def get_restore_id_func(	
	scale, shift, 
	slope_vg=0, thre_vg=0,
    slope_vd=0, thre_vd=0,
):
	def restore_id_func(ids, vg, vd):
		# ori_vg = vgs * scale['vg'] + shift
		# ori_vd = vds * scale['vd']
		ori_id = ids * scale['id'] / (
        (np.power(10, -slope_vg*(vg + thre_vg)) + 1) * 
        (np.power(10, -slope_vd*(abs(vd) - thre_vd)) + 1)
		) if slope_vg > 0 or slope_vd > 0 else ids * scale['id']
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

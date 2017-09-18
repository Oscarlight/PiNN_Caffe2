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
    	np.exp(-slope * (vg + threshold)) + 1
    	) if slope > 0 else ids/scale['id']

    return preproc_vg, preproc_vd, preproc_id

def get_restore_func(	
	scale, shift, 
	slope=0, threshold=0
):
	def restore_func(vg, vd, ids):
		ori_vg = vg * scale['vg'] + shift
    	ori_vd = vd * scale['vd']
    	ori_id = ids * scale['id'] / (
 		  	np.exp(-slope*(vg + threshold)) + 1
 			) if slope > 0 else ids * scale['id']
    	return ori_vg, ori_vd, ori_id
	return restore_func

def compute_dc_meta(vg, vd, ids):
    vg_shift = np.median(vg)-0.0
    vg_scale = max(abs(np.max(vg)-vg_shift)/1.0, abs(np.min(vg)-vg_shift)/1.0)
    vd_scale = max(abs(np.max(vd))/1.0, abs(np.min(vd))/1.0)
    id_scale = max(abs(np.max(ids))/0.85, abs(np.min(ids))/0.85)

    scale = {'vg':vg_scale, 'vd':vd_scale, 'id':id_scale}

    return scale, vg_shift

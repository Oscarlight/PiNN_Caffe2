import numpy as np
from scipy.integrate import simps

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
	
def compute_ac_meta(vg, vd, gradient, checkaccuracy = False):
    vg_shift = np.median(vg)-0.0
    vg_scale = max(abs(np.max(vg)-vg_shift)/1.0, abs(np.min(vg)-vg_shift)/1.0)
    vd_scale = max(abs(np.max(vd))/1.0, abs(np.min(vd))/1.0)
    integral, max_error, avg_error  = integrate(vg, vd, gradient, checkaccuracy)
    if (checkaccuracy):
        print("Max integration error: ", max_error)
        print("Average integration error: ", avg_error)
    q_scale = max(abs(np.max(integral))/0.75, abs(np.min(integral))/0.75)	
    scale = {'vg':vg_scale, 'vd':vd_scale, 'q':q_scale}
    return scale, vg_shift

def integrate(vg, vd, gradient, checkaccuracy):
    assert(len(vg) == len(vd)), "Fatal dimension mismatch"
    gradient = np.asarray(gradient)
    dqdvg = gradient[:, 0]
    dqdvd = gradient[:, 1]
    assert(len(dqdvg) == len(vg)), "Fatal dimension mismatch"
    assert(len(dqdvd) == len(vg)), "Fatal dimension mismatch"  
 
    #Determine dimensions of q matrix
    vg_x = True
    if (vg[0] == vg[1]):
	vg_x = False
        i = 1
	try:
	    while (vg[i] == vg[i-1]):
                i = i+1
            columns = i
        except IndexError:
	    print("Data not formatted well.")
	    columns = len(vg)
        i = 1
    else:
        i = 1
        try: 
            while (vd[i] == vd[i-1]):
                i=i+1
            columns = i
        except IndexError:
            print("Data not formatted well.")
            columns = len(vd)
    rows = len(vg)/columns
    
    #Reshape arrays and create dummy matrix 
    vd = np.reshape(vd, (rows, columns))
    vg = np.reshape(vg, (rows, columns))
    dqdvg = np.reshape(dqdvg, (rows, columns))
    dqdvd = np.reshape(dqdvd, (rows, columns))
    qs = np.zeros((rows, columns), dtype = np.float32)
    
    #Calculate integral using scipy's simps method
    if (vg_x): 
        for j in range(0, columns):
            for i in range(0, rows):
                if (i == 0):
                    if (j != 0):
                        qs[i,j] = qs[i, j-1] + simps(dqdvg[i,j-1:j+1], vg[i,j-1:j+1])
                else:
                    if (j == 0):
                        qs[i,j] = qs[i-1,j] + simps(dqdvd[i-1:i+1,j], vd[i-1:i+1,j])
                    else:
                        qs[i,j] = 0.5* (qs[i-1,j] + simps(dqdvd[i-1:i+1,j], vd[i-1:i+1,j])) + 0.5*(qs[i,j-1] + simps(dqdvg[i,j-1:j+1],vg[i,j-1:j+1]))
    else: 
      for j in range(0, columns):
            for i in range(0, rows):
                if (i == 0):
                    if (j != 0):
                        qs[i,j] = qs[i,j-1] + simps(dqdvd[i,j-1:j+1], vd[i,j-1:j+1])
                else:
                    if (j == 0):
                        qs[i,j] = qs[i-1,j] + simps(dqdvg[i-1:i+1,j], vg[i-1:i+1,j])
                    else:
                        qs[i,j] = 0.5*(qs[i,j-1] + simps(dqdvd[i,j-1:j+1],                                     vd[i,j-1:j+1])) + 0.5*(qs[i-1,j] + simps(                                       dqdvg[i-1:i+1,j], vg[i-1:i+1,j]))
    if (checkaccuracy):
        max_error, avg_error = checkAccuracy(qs, vg, vd, dqdvg, dqdvd, vg_x)
    else:
        max_error = 0
        avg_error = 0 
    return qs, max_error, avg_error

def checkAccuracy(qs, vg, vd, dqdvg, dqdvd, vg_x):
    dqdvg_calc = np.zeros(dqdvg.shape)
    dqdvd_calc = np.zeros(dqdvd.shape)
    if (vg_x): 
        x_axis = vg[0,:]
        y_axis = vd[:,0]
        gradient = np.gradient(qs, y_axis, x_axis)
        dqdvg_calc = gradient[1]
        dqdvd_calc = gradient[0]
    else:
        x_axis = vd[0,:]
        y_axis = vg[:,0]
        gradient = np.gradient(qs, y_axis, x_axis)
        dqdvg_calc = gradient[0]
        dqdvd_calc = gradient[1]
    (rows, columns) = dqdvg.shape
    max_error = 0
    avg_error = 0
    for j in range (0, columns): 
       for i in range (0, rows):
           dqdvg_error = abs((dqdvg_calc[i,j] - dqdvg[i, j])/ dqdvg[i,j])
           dqdvd_error = abs((dqdvd_calc[i,j] - dqdvd[i, j])/ dqdvd[i,j])
           if (dqdvg_error > max_error):
               max_error = dqdvg_error
           if (dqdvd_error > max_error):
               max_error = dqdvd_error
           print(dqdvg_error)
           print(dqdvd_error)
           avg_error = avg_error + dqdvd_error + dqdvg_error
    avg_error = avg_error/(rows*columns)    
    return max_error, avg_error   

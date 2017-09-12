import caffe2_paths
import numpy as np

from caffe2.python import (
	core, utils, workspace, schema, layer_model_helper
)
from caffe2.proto import caffe2_pb2

def write_db(db_type, db_name, sig_input, tanh_input, labels):
	''' The minidb datebase seems immutable.
	'''
	db = core.C.create_db(db_type, db_name, core.C.Mode.write)
	transaction = db.new_transaction()
	assert sig_input.shape == tanh_input.shape, 'two inputs have the same size.'
	for i in range(sig_input.shape[0]):
		tensor = caffe2_pb2.TensorProtos()
		tensor.protos.extend(
				[utils.NumpyArrayToCaffe2Tensor(sig_input[i]),
				 utils.NumpyArrayToCaffe2Tensor(tanh_input[i]),
				 utils.NumpyArrayToCaffe2Tensor(labels[i])])
		transaction.put(str(i), tensor.SerializeToString())
	del transaction
	del db

def add_input_only():
	''' for prediction
	'''
	pass

def add_input_and_label(
	model, db_name, db_type, batch_size=1
):
	assert batch_size != 0, 'batch_size cannot be zero'
	reader_init_net = core.Net('reader_init_net')
	dbreader = reader_init_net.CreateDB(
		[], "dbreader", db=db_name, db_type=db_type)
	workspace.RunNetOnce(reader_init_net) # need to initialze dbreader ONLY ONCE
	sig_input, tanh_input, label = model.TensorProtosDBInput(
		[dbreader], 
		["sig_input", "tanh_input", "label"], 
		batch_size=batch_size
	)
	sig_input = model.StopGradient(sig_input, sig_input)
	tanh_input = model.StopGradient(tanh_input, tanh_input)
	model.input_feature_schema.sig_input.set_value(sig_input.get(), unsafe=True)
	model.input_feature_schema.tanh_input.set_value(tanh_input.get(), unsafe=True)
	model.trainer_extra_schema.label.set_value(label.get(), unsafe=True)
	return sig_input, tanh_input, label

# @ Xiang: please implement this function by 09/12
#          Note: please add unittest in preproc_test.py

def dc_iv_preproc(vg, vd, id, scale, shift):
    '''
    inputs:
        1) two numpy array features and labels
        2) arguments for pre-processing
    outputs:
        1) pre-process features and labels
        2) a function for restoring to the original data

    '''
    preproc_vg = (vg-shift)/scale['vg']
    preproc_vd = vd/scale['vd']
    preproc_id = id/scale['id']

    def restore_func(vg, vd, id, scale, shift):
        ori_vg = vg*scale['vg']+shift
        ori_vd = vd*scale['vd']
        ori_id = id*scale['id']
        return ori_vg, ori_vd, ori_id

    return preproc_vg, preproc_vd, preproc_id, restore_func

def compute_meta(vg, vd, id):

    vg_shift = np.median(vg)-0.0
    vg_scale = max(abs(np.max(vg)-vg_shift)/1.0, abs(np.min(vg)-vg_shift)/1.0)
    vd_scale = max(abs(np.max(vd))/1.0, abs(np.min(vd))/1.0)
    id_scale = max(abs(np.max(id))/0.9, abs(np.min(id))/0.9)

    scale = {'vg':vg_scale, 'vd':vd_scale, 'id':id_scale}
    shift = vg_shift

    return scale, shift

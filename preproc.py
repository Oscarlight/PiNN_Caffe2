import caffe2_paths
import numpy as np
import math

from caffe2.python import (
	core, utils, workspace, schema, layer_model_helper
)
from caffe2.proto import caffe2_pb2

def write_db(db_type, db_name, input_1, input_2, labels):
	''' The minidb datebase seems immutable.
	'''
	db = core.C.create_db(db_type, db_name, core.C.Mode.write)
	transaction = db.new_transaction()
	assert input_1.shape == input_2.shape, 'two inputs have the same size.'
	for i in range(input_1.shape[0]):
		tensor = caffe2_pb2.TensorProtos()
		tensor.protos.extend(
				[utils.NumpyArrayToCaffe2Tensor(input_1[i]),
				 utils.NumpyArrayToCaffe2Tensor(input_2[i]),
				 utils.NumpyArrayToCaffe2Tensor(labels[i])])
		transaction.put(str(i), tensor.SerializeToString())
	del transaction
	del db

def add_input_only():
	''' for prediction
	'''
	pass

def add_two_inputs_and_label(
	model, 
	db_name, db_type, 
	input_1_name, input_2_name,
	batch_size=1
):
	'''
	For adjoint MLP and Pi-NN
	'''
	assert batch_size != 0, 'batch_size cannot be zero'
	reader_init_net = core.Net('reader_init_net')
	dbreader = reader_init_net.CreateDB(
		[], "dbreader", db=db_name, db_type=db_type)
	workspace.RunNetOnce(reader_init_net) # need to initialze dbreader ONLY ONCE
	input_1, input_2, label = model.TensorProtosDBInput(
		[dbreader], 
		[input_1_name, input_2_name, "label"], 
		batch_size=batch_size
	)
	input_1 = model.StopGradient(input_1, input_1)
	input_2 = model.StopGradient(input_2, input_2)
	# set inputs and label
	model.input_feature_schema.input_1.set_value(input_1.get(), unsafe=True)
	model.input_feature_schema.input_2.set_value(input_2.get(), unsafe=True)
	model.trainer_extra_schema.label.set_value(label.get(), unsafe=True)
	return input_1, input_2, label

def add_three_inputs_and_label(
):
	'''
	For adjoint Pi-NN
	'''
	pass

def dc_iv_preproc(vg, vd, ids, scale, shift, 
	slope = 0, threshold = 0
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
    preproc_id_scalar = [(math.exp(-slope * (x + threshold)) + 1) for x in vg]
    preproc_id = ids / scale['id'] * preproc_id_scalar if slope > 0 else ids/scale['id']

    def restore_func(vg, vd, ids):
        ori_vg = vg * scale['vg'] + shift
        ori_vd = vd * scale['vd']
        ori_id_scalar = [(math.exp(-slope * (x + threshold)) + 1) for x in vg]
        ori_id = ids * scale['id'] / ori_id_scalar if slope > 0 else ids * scale['id']
        return ori_vg, ori_vd, ori_id

    return preproc_vg, preproc_vd, preproc_id, restore_func

def compute_dc_meta(vg, vd, ids):

    vg_shift = np.median(vg)-0.0
    vg_scale = max(abs(np.max(vg)-vg_shift)/1.0, abs(np.min(vg)-vg_shift)/1.0)
    vd_scale = max(abs(np.max(vd))/1.0, abs(np.min(vd))/1.0)
    id_scale = max(abs(np.max(ids))/0.85, abs(np.min(ids))/0.85)

    scale = {'vg':vg_scale, 'vd':vd_scale, 'id':id_scale}

    return scale, vg_shift

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
def dc_iv_preproc(features, labels, scale = 2.0):
	'''
	input: 
	    1) two numpy array features and labels
	    2) arguments for preprocessing
	output: 
		1) preprocss features and labels
		2) a function for restoring to the origin data
	'''
	preproc_features = features
	preproc_labels = labels
	restore_func = None
	return preproc_features, preproc_labels, restore_func

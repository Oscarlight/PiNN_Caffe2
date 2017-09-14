import caffe2_paths
import numpy as np

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

def add_input_and_label(
	model, 
	db_name, db_type, 
	input_1_name, input_2_name,
	batch_size=1
):
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

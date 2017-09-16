import numpy as np
import math
from caffe2.python import (
	core, utils, workspace, schema, layer_model_helper
)
from caffe2.proto import caffe2_pb2
from caffe2.python.layers.tags import Tags

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
				 utils.NumpyArrayToCaffe2Tensor(labels[i])]
		)
		transaction.put(str(i), tensor.SerializeToString())
	del transaction
	del db
	return db_name

def build_input_reader(
	model,
	db_name, db_type,
	input_names_lst,
	batch_size = 1,
	data_type='train',
):
	'''
	Init the dbreader and build the network for reading the data,
	however, the newwork is not connected to the computation network yet.
	Therefore we can switch between different data sources.
	'''
	assert batch_size != 0, 'batch_size cannot be zero'
	reader_init_net = core.Net('reader_init_net_'+data_type)
	dbreader = reader_init_net.CreateDB(
		[], "dbreader", db=db_name, db_type=db_type)
	# need to initialze dbreader ONLY ONCE
	workspace.RunNetOnce(reader_init_net)
	if data_type == 'train':
		TAG = Tags.TRAIN_ONLY
	elif data_type == 'eval':
		TAG = Tags.EVAL_ONLY
	else:
		raise Exception('data type: {} not valid.'.format(data_type))
	with Tags(TAG):
		# the last one is the label
		input_data_struct = model.TensorProtosDBInput(
			[dbreader], 
			input_names_lst + ["label"],
			name = 'DBInput_' + data_type,
			batch_size=batch_size
		)
		input_data_lst = [input_data for input_data in input_data_struct]
		for i in range(len(input_data_lst)-1):
			input_data_lst[i] = model.StopGradient(
				input_data_lst[i], input_data_lst[i]
			)
	return input_data_lst
import caffe2_paths
import numpy as np

from caffe2.python import core, utils, workspace, schema
from caffe2.proto import caffe2_pb2

def write_db(db_type, db_name, features, labels):
	db = core.C.create_db(db_type, db_name, core.C.Mode.write)
	transaction = db.new_transaction()
	for n in range(features.shape[0]):
		tensor = caffe2_pb2.TensorProtos()
		tensor.protos.extend([utils.NumpyArrayToCaffe2Tensor(features[n]),
				utils.NumpyArrayToCaffe2Tensor(labels[n])])
		transaction.put(str(n), tensor.SerializeToString())
	del transaction
	del db


def add_input(model, batch_size, db, db_type):
	feature, label = model.TensorProtosDBInput(
			[], ["feature", "label"],
			batch_size = batch_size, 
			db=db, db_type=db_type
	)
	schema.FeedRecord(model.input_feature_schema, feature)
	schema.FeedRecord(model.trainer_extra_schema, label)
	return data, label

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

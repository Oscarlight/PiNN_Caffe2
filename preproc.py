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

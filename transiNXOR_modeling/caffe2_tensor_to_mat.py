import sys
sys.path.append('../')
import caffe2_paths
from caffe2.python import workspace
from pinn import exporter
import numpy as np
from scipy.io import savemat
import pickle
import glob

model_name = 'bise_ext_sym_h264_0'
dest_folder = 'c_model/'

init_net = exporter.load_init_net('./transiXOR_Models/'+model_name+'_init')
print(type(init_net))
saved_mat = {}
for op in init_net.op:
	tensor = workspace.FetchBlob(op.output[0])
	tensor_name = op.output[0].replace('/', '_')
	print(tensor_name)
	saved_mat[tensor_name] = tensor
savemat(dest_folder + 'params.mat', saved_mat)

## Preprocess param
saved_preproc = {}
with open("./transiXOR_Models/"+model_name+"_preproc_param.p", "rb") as f:
	preproc_dict = pickle.load(f)
saved_preproc = preproc_dict['scale']
saved_preproc['vg_shift'] = preproc_dict['vg_shift']
print(saved_preproc)
savemat(dest_folder + 'preproc.mat', saved_preproc)

ids_file = glob.glob('./transiXOR_data/current_D9.npy')
ids_data = np.load(ids_file[0])
savemat(dest_folder + 'current.mat', {'data': ids_data})

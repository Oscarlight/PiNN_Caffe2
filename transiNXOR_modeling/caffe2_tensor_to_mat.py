import sys
sys.path.append('../')
import caffe2_paths
from caffe2.python import workspace
from pinn import exporter
from scipy.io import savemat
import pickle

model_name = 'bise_h216_0'

init_net = exporter.load_init_net('./transiXOR_Models/'+model_name+'_init')
print(type(init_net))
saved_mat = {}
for op in init_net.op:
	tensor = workspace.FetchBlob(op.output[0])
	tensor_name = op.output[0].replace('/', '_')
	print(tensor_name)
	saved_mat[tensor_name] = tensor
savemat('params.mat', saved_mat)

## Preprocess param
saved_preproc = {}
with open("./transiXOR_Models/"+model_name+"_preproc_param.p", "rb") as f:
	preproc_dict = pickle.load(f)
saved_preproc = preproc_dict['scale']
saved_preproc['vg_shift'] = preproc_dict['vg_shift']
print(saved_preproc)
savemat('preproc.mat', saved_preproc)
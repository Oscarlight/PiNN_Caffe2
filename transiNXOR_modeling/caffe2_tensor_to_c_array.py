import sys
sys.path.append('../')
import caffe2_paths
from caffe2.python import workspace
from pinn import exporter
from scipy.io import savemat
import numpy as np
import pickle

model_name = 'bise_h216_0'

init_net = exporter.load_init_net('./transiXOR_Models/'+model_name+'_init')
print(type(init_net))
p = {}
with open("c_model/c_arrays.txt","w") as f:
	f.write('/* --------------- MODEL: '+ model_name +' -------------------- */')
	for op in init_net.op:
		tensor = workspace.FetchBlob(op.output[0])
		tensor_name = op.output[0].replace('/', '_')
		print(tensor_name)
		print(tensor.shape)
		p[tensor_name] = tensor
		tensor_str = np.array2string(tensor.flatten(), separator=',')
		tensor_str = tensor_str.replace("[", "{").replace("]", "}")
		str = 'float ' + tensor_name + '[] = ' + tensor_str + ';\n'
		f.write(str)

	## Preprocess param
	with open("./transiXOR_Models/"+model_name+"_preproc_param.p", "rb") as f:
		preproc_dict = pickle.load(f)
	print(preproc_dict)

## TESTING
vg = np.array([0.2, 0.2]); vd = np.array([0.2])
vg = (vg - 0.1)/0.1; vd /= 0.2
tanh_temp0 = p['tanh_fc_layer_0_w']*vd
sig_temp0 = p['sig_fc_layer_0_w'].dot(vg) + p['sig_fc_layer_0_b']
print(sig_temp0)



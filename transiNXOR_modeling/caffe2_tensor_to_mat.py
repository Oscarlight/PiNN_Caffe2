import sys
sys.path.append('../')
import caffe2_paths
from caffe2.python import workspace
from pinn import exporter
from scipy.io import savemat

init_net = exporter.load_init_net('./transiXOR_Models/model_output_0_init')
print(type(init_net))
saved_mat = {}
for op in init_net.op:
	tensor = workspace.FetchBlob(op.output[0])
	saved_mat[op.output[0]] = tensor
savemat('params.mat', saved_mat)
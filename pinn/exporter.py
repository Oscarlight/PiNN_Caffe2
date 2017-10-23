import caffe2_paths

import numpy as np
from caffe2.python import workspace, core, model_helper, brew, optimizer, utils
from caffe2.proto import caffe2_pb2


def load_net(INIT_NET, PREDICT_NET):

	init_def = caffe2_pb2.NetDef()
	with open(INIT_NET+'.model', 'r') as f:
	    init_def.ParseFromString(f.read())
	    #init_def.device_option.CopyFrom(device_opts)
	    workspace.RunNetOnce(init_def.SerializeToString())
	    #print(init_def)

	net_def = caffe2_pb2.NetDef()
	with open(PREDICT_NET+'.model', 'r') as f:
	    net_def.ParseFromString(f.read())
	    workspace.CreateNet(net_def.SerializeToString(), overwrite=True)
	    #print(net_def)

	# return net_def.SerializeToString()
	return net_def.name


def save_net(net, model, INIT_NET, PREDICT_NET):

    with open(PREDICT_NET+'.model', 'wb') as f:
        f.write(net._net.SerializeToString())
    init_net = caffe2_pb2.NetDef()
    for param in model.get_parameter_blobs():
        blob = workspace.FetchBlob(param)
        shape = blob.shape
        op = core.CreateOperator("GivenTensorFill", [], [param],arg=[ utils.MakeArgument("shape", shape),utils.MakeArgument("values", blob)])
        init_net.op.extend([op])
    with open(INIT_NET+'.model', 'wb') as f:
        f.write(init_net.SerializeToString())
import numpy as np
import preproc
from caffe2.python import workspace, layer_model_helper, schema, core, utils, workspace

features = np.random.rand(10, 2).astype('float32')
labels = np.random.rand (10, 2).astype('float32')

workspace.ResetWorkspace()
preproc.write_db('minidb', 'test.db', features, labels)
net_proto = core.Net("example_reader")
dbreader = net_proto.CreateDB([], "dbreader", db="test.db", db_type="minidb")
net_proto.TensorProtosDBInput([dbreader], ["data", "label"], batch_size=10)

print("The net looks like this:")
print(str(net_proto.Proto()))

workspace.CreateNet(net_proto)
workspace.RunNet(net_proto.Proto().name)

print("Features after running net once:")
print(workspace.FetchBlob("data"))
print("\nLabels after running net once:")
print(workspace.FetchBlob("label"))

workspace.RunNet(net_proto.Proto().name)

print("\nFeatures after running net twice:")
print(workspace.FetchBlob("data"))
print("\nLabels after running net twice:")
print(workspace.FetchBlob("label"))

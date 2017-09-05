import sys
if sys.platform == 'macox':
	sys.path.append("/Users/Mingda/Documents/Caffe2/caffe2/build/")
if sys.platform == 'linux2':
	sys.path.append("/home/oscar/Documents/caffe2/build/")

from caffe2.python import (
	workspace, layer_model_helper, schema, optimizer, net_drawer
)
import caffe2.python.layer_model_instantiator as instantiator 
import numpy as np
from pinn_lib import build_pinn

workspace.ResetWorkspace()
input_record_schema = schema.Struct(
		('sig_input', schema.Scalar((np.float32, (2, )))),
		('tanh_input', schema.Scalar((np.float32, (2, ))))
	)
output_record_schema = schema.Struct(
		('loss', schema.Scalar((np.float32, (2, )))),
		('pred', schema.Scalar((np.float32, (2, ))))
	)
trainer_extra_schema = schema.Struct()
model = layer_model_helper.LayerModelHelper(
	"pinn_example",
	input_record_schema,
	trainer_extra_schema)
# example data
X_sig = np.array([[2., 2.], [2., 2.], [3., 4.]], dtype = np.float32)
Y_tanh = np.array([[1., 1.], [1., 1.], [2., 5.]], dtype = np.float32)
label = np.ones((3, 2), dtype = np.float32)
schema.FeedRecord(model.input_feature_schema, [X_sig, Y_tanh])
# build the model
pred, loss = build_pinn(
	model,
	label,
	2,
	sig_net_dim = [3, 2],
	tanh_net_dim = [5, 2],
	inner_embed_dim = [2, 3],
	optim=optimizer.AdagradOptimizer()
)
model.add_loss(loss)
output_record = schema.NewRecord(
	model.net,
	output_record_schema
)
output_record_schema.pred.set_value(pred.get(), unsafe=True)
output_record_schema.loss.set_value(loss.get(), unsafe=True)
model.output_schema = output_record_schema

train_init_net, train_net = instantiator.generate_training_nets(model)

# Train the model
workspace.RunNetOnce(train_init_net)

# graph = net_drawer.GetPydotGraph(train_net.Proto().op, rankdir='TB')
# with open(train_net.Name() + ".png",'wb') as f:
# 	f.write(graph.create_png())

workspace.CreateNet(train_net)
num_iter = 1000
eval_num_iter = 2
for i in range(eval_num_iter):
	workspace.RunNet(train_net.Proto().name, num_iter=num_iter)
	print(schema.FetchRecord(loss))
	print(schema.FetchRecord(pred))

# # Eval
# X_sig = np.array([[2., 2.], [2., 2.], [3., 4.]], dtype = np.float32)
# Y_tanh = np.array([[1., 1.], [1., 1.], [2., 5.]], dtype = np.float32)
# label = np.array([[0.], [1.], [3.]], dtype = np.float32)
# schema.FeedRecord(model.input_feature_schema, [X_sig, Y_tanh])
# eval_net = instantiator.generate_eval_net(model)
# # graph = net_drawer.GetPydotGraph(eval_net.Proto().op, rankdir='TB')
# # with open(eval_net.Name() + ".png",'wb') as f:
# # 	f.write(graph.create_png())
# workspace.CreateNet(eval_net)
# workspace.RunNet(eval_net.Proto().name)
# print(schema.FetchRecord(loss))
# print(schema.FetchRecord(pred))

# # Predict1
# pred_net = instantiator.generate_predict_net(model)
# graph = net_drawer.GetPydotGraph(pred_net.Proto().op, rankdir='TB')
# with open(pred_net.Name() + ".png",'wb') as f:
# 	f.write(graph.create_png())
# workspace.CreateNet(pred_net)
# workspace.RunNet(pred_net.Proto().name)
# print(schema.FetchRecord(pred))
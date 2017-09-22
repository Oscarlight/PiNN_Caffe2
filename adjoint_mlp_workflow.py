import caffe2_paths
import os
from caffe2.python import (
	core, workspace, layer_model_helper, schema, optimizer, net_drawer
)
import caffe2.python.layer_model_instantiator as instantiator
from pinn.data_reader import write_db, build_input_reader
from pinn.adjoint_mlp_lib import (
	build_adjoint_mlp, init_model_with_schemas
)
import numpy as np

input_dim = 1
output_dim = 1
hidden_dims = [10, 10, 10]
model = init_model_with_schemas('adjoint_nn', input_dim, output_dim)
# example data
# TODO: use preproc
x_array = np.linspace(-0.8, 0.8, 100)
origin_input = np.array([[x] for x in x_array], dtype = np.float32)
adjoint_label = np.array(
	[[(1 - np.exp(10*x)) / (1 + np.exp(10*x))] for x in x_array], 
	dtype = np.float32
)
adjoint_input = np.ones((100,1), dtype = np.float32)
# create db if needed
db_name = 'adjoint_nn.db'
if not os.path.isfile(db_name):
	print(">>> Create a new database...")
	write_db('minidb', db_name, 
		origin_input, adjoint_input, adjoint_label)
else:
	print(">>> The database with the same name already existed.")
origin_input, adjoint_input, label = build_input_reader(
	model, db_name, 'minidb', ['origin_input', 'adjoint_input'], batch_size=100
)
model.input_feature_schema.origin_input.set_value(
	origin_input.get(), unsafe=True)
model.input_feature_schema.adjoint_input.set_value(
	adjoint_input.get(), unsafe=True)
model.trainer_extra_schema.label.set_value(
	label.get(), unsafe=True)
# Build model
origin_pred, adjoint_pred, loss = build_adjoint_mlp(
	model, 
	input_dim=input_dim,
	hidden_dims=hidden_dims,
	output_dim=output_dim,
	optim=optimizer.AdagradOptimizer(alpha=0.01, epsilon=1e-4,))

# Train the model
train_init_net, train_net = instantiator.generate_training_nets(model)
workspace.RunNetOnce(train_init_net)
workspace.CreateNet(train_net)
num_iter = 10000
eval_num_iter = 1
for i in range(eval_num_iter):
	workspace.RunNet(train_net.Proto().name, num_iter=num_iter)
	print(schema.FetchRecord(loss).get())
import matplotlib.pyplot as plt
origin_pred_array = np.squeeze(schema.FetchRecord(origin_pred).get())
plt.plot(x_array, np.gradient(origin_pred_array, np.squeeze(x_array)), 'r')
plt.plot(x_array, origin_pred_array, 'r')
plt.plot(x_array, schema.FetchRecord(adjoint_pred).get(), 'b')
plt.plot(x_array, adjoint_label, 'b--')
plt.show() 
# Eval
eval_net = instantiator.generate_eval_net(model)
graph = net_drawer.GetPydotGraph(eval_net.Proto().op, rankdir='TB')
with open(eval_net.Name() + ".png",'wb') as f:
	f.write(graph.create_png())
# Predict1
# pred_net = instantiator.generate_predict_net(model)
# graph = net_drawer.GetPydotGraph(pred_net.Proto().op, rankdir='TB')
# with open(pred_net.Name() + ".png",'wb') as f:
# 	f.write(graph.create_png())
# origin_input = np.array([[0.0]], dtype=np.float32 )
# adjoint_input = np.array([[1.0]], dtype=np.float32 )
# schema.FeedRecord(model.input_feature_schema, [origin_input, adjoint_input])
# workspace.CreateNet(pred_net)
# workspace.RunNet(pred_net.Proto().name)
# print(schema.FetchRecord(origin_pred))
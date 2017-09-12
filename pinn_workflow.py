import caffe2_paths
import os
from caffe2.python import (
	workspace, layer_model_helper, schema, optimizer, net_drawer
)
import caffe2.python.layer_model_instantiator as instantiator 
import numpy as np
from pinn_lib import build_pinn, init_model_with_schemas
from preproc import write_db, add_input_and_label
# workspace.ResetWorkspace()
model = init_model_with_schemas('pinn_example', 2, 2, 2)
# example data
sig_input = np.array([[2., 1.], [2., 2.], [3., 4.]], dtype = np.float32)
tanh_input = np.array([[1., 1.], [1., 2.], [2., 5.]], dtype = np.float32)
label = np.ones((3, 2), dtype = np.float32)
# create db if needed
db_name = 'pinn.db'
if not os.path.isfile(db_name):
	print("Create a new database...")
	write_db('minidb', db_name, 
		sig_input, tanh_input, label)
else:
	print("The database with the same name already existed.")
sig_input, tanh_input, label = add_input_and_label(
	model, db_name, 'minidb', 'sig_input', 'tanh_input', batch_size=1
)
# build the model
pred, loss = build_pinn(
	model,
	label,
	sig_net_dim = [3, 2],
	tanh_net_dim = [5, 2],
	inner_embed_dim = [2, 3],
	optim=optimizer.AdagradOptimizer(alpha=0.01, epsilon=1e-4,)
)
# Train the model
train_init_net, train_net = instantiator.generate_training_nets(model)
workspace.RunNetOnce(train_init_net)
workspace.CreateNet(train_net)
num_iter = 1000
eval_num_iter = 4
for i in range(eval_num_iter):
	print('--------')
	workspace.RunNet(train_net, num_iter=num_iter)
	# print(schema.FetchRecord(tanh_input).get())
	print(schema.FetchRecord(loss).get())
	# print(schema.FetchRecord(label).get())
	# print(schema.FetchRecord(pred).get())

# # Eval
# X_sig = np.array([[2., 2.], [2., 2.], [3., 4.]], dtype = np.float32)
# Y_tanh = np.array([[1., 1.], [1., 1.], [2., 5.]], dtype = np.float32)
# label = np.array([[0.], [1.], [3.]], dtype = np.float32)
# schema.FeedRecord(model.input_feature_schema, [X_sig, Y_tanh])
eval_net = instantiator.generate_eval_net(model)
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
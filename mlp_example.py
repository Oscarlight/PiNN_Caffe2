import sys
if sys.platform == 'darwin':
	sys.path.append("/Users/Mingda/Documents/Caffe2/caffe2/build/")
if sys.platform == 'linux2':
	sys.path.append("/home/oscar/Documents/caffe2/build/")

from caffe2.python import (
	core, workspace, layer_model_helper, schema, optimizer, net_drawer, scope
)
from caffe2.python.modeling.parameter_sharing import (
    ParameterSharing,
)
import caffe2.python.layer_model_instantiator as instantiator 
import numpy as np

def build_mlp(
	model,
	label,
	input_dim = 1,
	hidden_dims = 2,
	output_dim = 1,
	optim=None,
):
	gamma1 = model.FCWithoutBias(
		model.input_feature_schema.input, 
		hidden_dims, 
		weight_optim=optim,
		name='fc1'
	)
	z1 = model.Sigmoid(gamma1, 'sig1')
	gamma2 = model.FCWithoutBias(
		z1, 
		hidden_dims,
		weight_optim=optim, 
		name='fc2'
	)
	z2 = model.Sigmoid(gamma2, 'sig2')
	gamma3 = model.FCWithoutBias(
		z2, 
		output_dim,
		weight_optim=optim, 
		name='fc3'
	)
	pred = gamma3
	loss_input_record = schema.NewRecord(
		model.net,
		schema.Struct(
			('label', schema.Scalar((np.float32, (output_dim, )))),
			('prediction', schema.Scalar((np.float32, (output_dim, ))))
		)
	)
	schema.FeedRecord(loss_input_record.label, [label])
	loss_input_record.prediction.set_value(pred.get(), unsafe=True)
	loss = model.BatchDirectMSELoss(loss_input_record)
	return pred, loss

if __name__ == '__main__':
	workspace.ResetWorkspace()
	input_dim = 1
	output_dim = 1
	hidden_dims = 10
	input_record_schema = schema.Struct(
			('input', schema.Scalar((np.float32, (input_dim, )))),
		)
	output_record_schema = schema.Struct(
			('loss', schema.Scalar((np.float32, (input_dim, )))),
			('pred', schema.Scalar((np.float32, (output_dim, ))))
		)
	trainer_extra_schema = schema.Struct()
	model = layer_model_helper.LayerModelHelper(
		"pinn_example",
		input_record_schema,
		trainer_extra_schema)
	# example data
	input = np.array([[e] for e in np.linspace(0.3, 0.8, 100)], dtype = np.float32)
	label = input
	schema.FeedRecord(model.input_feature_schema, [input])

	pred, loss = build_mlp(
		model, 
		label,
		input_dim=input_dim,
		hidden_dims=hidden_dims,
		output_dim=output_dim,
		optim=optimizer.AdagradOptimizer())

	model.add_loss(loss)
	output_record = schema.NewRecord(
		model.net,
		output_record_schema
	)
	output_record_schema.pred.set_value(pred.get(), unsafe=True)
	output_record_schema.loss.set_value(loss.get(), unsafe=True)
	model.output_schema = output_record_schema

	## =========================
	# Train the model
	train_init_net, train_net = instantiator.generate_training_nets(model)
	workspace.RunNetOnce(train_init_net)

	# graph = net_drawer.GetPydotGraph(train_net.Proto().op, rankdir='TB')
	# with open(train_net.Name() + ".png",'wb') as f:
	# 	f.write(graph.create_png())

	workspace.CreateNet(train_net)
	num_iter = 100000
	eval_num_iter = 1
	for i in range(eval_num_iter):
		workspace.RunNet(train_net.Proto().name, num_iter=num_iter)
		print(schema.FetchRecord(loss).get())
	import matplotlib.pyplot as plt
	plt.plot(label, 'r--')
	plt.plot(schema.FetchRecord(pred).get(), 'r')
	plt.show()


	# # Predict1
	# pred_net = instantiator.generate_predict_net(model)
	# graph = net_drawer.GetPydotGraph(pred_net.Proto().op, rankdir='TB')
	# with open(pred_net.Name() + ".png",'wb') as f:
	# 	f.write(graph.create_png())
	# # workspace.CreateNet(pred_net)
	# # workspace.RunNet(pred_net.Proto().name)
	# # print(schema.FetchRecord(pred))
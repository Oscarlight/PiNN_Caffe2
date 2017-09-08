import caffe2_paths

from caffe2.python import (
	core, workspace, layer_model_helper, schema, optimizer, net_drawer, scope
)
from caffe2.python.modeling.parameter_sharing import (
    ParameterSharing,
)
import caffe2.python.layer_model_instantiator as instantiator 
import numpy as np

def build_adjoint_mlp(
	model,
	label,
	input_dim = 1,
	hidden_dims = [2, 2],
	output_dim = 1,
	optim=None,
):
	assert len(hidden_dims) >= 1, "at least one hidden dim"
	with ParameterSharing({'origin' : 'adjoint'}):
		with scope.NameScope('origin'):
			gamma = model.FCWithoutBias(
				model.input_feature_schema.origin_input, 
				hidden_dims[0], 
				weight_optim=optim,
				name='fc1'
			)
			z = model.Sigmoid(gamma, 'sig1')
			z_lst = [z]
			for hidden_dim in hidden_dims[1:]:
				gamma = model.FCWithoutBias(
					z,
					hidden_dim, 
					weight_optim=optim,
					name='fc{}'.format(len(z_lst) + 1)
				)
				z = model.Sigmoid(gamma, 'sig{}'.format(len(z_lst) + 1))
				z_lst.append(z)
			origin_pred = model.FCWithoutBias(
				z_lst[-1], 
				output_dim,
				weight_optim=optim, 
				name='fc{}'.format(len(z_lst) + 1)
			)
		with scope.NameScope('adjoint'):
			idx = len(z_lst) + 1
			# print(idx)
			gamma_ad = model.FCTransposeW(
				model.input_feature_schema.adjoint_input, 
				hidden_dims[-1], 
				weight_optim=optim,
				name='fc{}'.format(idx)
			)
			z = z_lst.pop()
			one_vector = model.ConstantFill(
				[z], 
				'ones{}'.format(idx), 
				value=1.0, 
				dtype=core.DataType.FLOAT
			)
			multiplier = model.Mul(
				[z, model.Sub([one_vector, z], 'sub{}'.format(idx))],
					'multiplier{}'.format(idx),
			)
			# model.StopGradient(z, z)
			alpha = model.Mul([gamma_ad, multiplier], 'adjoint_layer{}'.format(idx))
			# one_vector = model.add_global_constant(name='ONES', array=np.ones(hidden_dims), dtype=np.float32)
			# one_vector = model.Cast(one_vector, 'FLOAT_ONES', dtype=core.DataType.FLOAT)
			while len(z_lst) > 0:
				idx = len(z_lst) + 1
				gamma = model.FCTransposeW(
					alpha, 
					hidden_dims[len(z_lst) - 1],
					weight_optim=optim,
					name='fc{}'.format(idx)
				)
				print(idx)
				z = z_lst.pop()
				one_vector = model.ConstantFill(
					[z],
					'ones{}'.format(idx),
					value=1.0, 
					dtype=core.DataType.FLOAT
				)
				multiplier = model.Mul(
					[z, model.Sub([one_vector, z], 'sub{}'.format(idx))],
						'multiplier{}'.format(idx),
				)
				# model.StopGradient(z, z)
				alpha = model.Mul([gamma_ad, multiplier], 'adjoint_layer{}'.format(idx))
			adjoint_pred = model.FCTransposeW(
				alpha, 
				input_dim,
				weight_optim=optim,
				name='fc1'
			)

	loss_input_record = schema.NewRecord(
		model.net,
		schema.Struct(
			('label', schema.Scalar((np.float32, (input_dim, )))),
			('prediction', schema.Scalar((np.float32, (input_dim, ))))
		)
	)
	schema.FeedRecord(loss_input_record.label, [label])
	loss_input_record.prediction.set_value(adjoint_pred.get(), unsafe=True)
	loss = model.BatchDirectMSELoss(loss_input_record)
	return origin_pred, adjoint_pred, loss

if __name__ == '__main__':
	workspace.ResetWorkspace()
	input_dim = 1
	output_dim = 1
	hidden_dims = [10, 10]
	input_record_schema = schema.Struct(
			('origin_input', schema.Scalar((np.float32, (input_dim, )))),
			('adjoint_input', schema.Scalar((np.float32, (output_dim, ))))
		)
	output_record_schema = schema.Struct(
			('loss', schema.Scalar((np.float32, (input_dim, )))),
			('origin_pred', schema.Scalar((np.float32, (output_dim, )))),
			('adjoint_pred', schema.Scalar((np.float32, (input_dim, ))))
		)
	trainer_extra_schema = schema.Struct()
	model = layer_model_helper.LayerModelHelper(
		"pinn_example",
		input_record_schema,
		trainer_extra_schema)
	# example data
	origin_input = np.array([[e] for e in np.linspace(0.3, 0.8, 100)], dtype = np.float32)
	adjoint_input = np.ones((100,1), dtype = np.float32)
	adjoint_label = origin_input
	schema.FeedRecord(model.input_feature_schema, [origin_input, adjoint_input])

	origin_pred, adjoint_pred, loss = build_adjoint_mlp(
		model, 
		adjoint_label,
		input_dim=input_dim,
		hidden_dims=hidden_dims,
		output_dim=output_dim,
		optim=optimizer.AdagradOptimizer())

	model.add_loss(loss)
	output_record = schema.NewRecord(
		model.net,
		output_record_schema
	)
	output_record_schema.origin_pred.set_value(origin_pred.get(), unsafe=True)
	output_record_schema.adjoint_pred.set_value(adjoint_pred.get(), unsafe=True)
	output_record_schema.loss.set_value(loss.get(), unsafe=True)
	model.output_schema = output_record_schema

	## =========================
	# # print(model.param_to_optim)
	# train_init_net, train_net = instantiator._generate_training_net_only(model)
	# grad_map = train_net.AddGradientOperators(model.loss.field_blobs())
	# # print(grad_map)
	# from future.utils import viewitems
	# for param, optimizer in viewitems(model.param_to_optim):
	# 	print(param)
	# 	print(grad_map.get(str(param)))

	# Train the model
	train_init_net, train_net = instantiator.generate_training_nets(model)
	workspace.RunNetOnce(train_init_net)

	# graph = net_drawer.GetPydotGraph(train_net.Proto().op, rankdir='TB')
	# with open(train_net.Name() + ".png",'wb') as f:
	# 	f.write(graph.create_png())

	workspace.CreateNet(train_net)
	num_iter = 5000
	eval_num_iter = 2
	for i in range(eval_num_iter):
		workspace.RunNet(train_net.Proto().name, num_iter=num_iter)
		print(schema.FetchRecord(loss).get())
	import matplotlib.pyplot as plt
	plt.plot(schema.FetchRecord(origin_pred).get(), 'r')
	plt.plot(schema.FetchRecord(adjoint_pred).get(), 'b')
	plt.plot(adjoint_label, 'b--')
	plt.show()
	# # Predict1
	# pred_net = instantiator.generate_predict_net(model)
	# graph = net_drawer.GetPydotGraph(pred_net.Proto().op, rankdir='TB')
	# with open(pred_net.Name() + ".png",'wb') as f:
	# 	f.write(graph.create_png())
	# # workspace.CreateNet(pred_net)
	# # workspace.RunNet(pred_net.Proto().name)
	# # print(schema.FetchRecord(pred))
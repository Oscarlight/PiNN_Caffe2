import caffe2_paths

from caffe2.python import (
	core, schema, optimizer, net_drawer, workspace, layer_model_helper
)
import numpy as np

def build_block(
	model,
	sig_input, tanh_input,
	sig_n, tanh_n,
	block_index,
	weight_optim=None,
	bias_optim=None,
	linear_activation = False
):
	tanh_h = model.FCWithoutBias(
		tanh_input, 
		tanh_n,
		weight_optim=weight_optim,
		name = 'tanh_fc_layer_{}'.format(block_index),
	)
	sig_h = model.FC(
		sig_input,
		sig_n,
		weight_optim=weight_optim,
		bias_optim=bias_optim,
		name = 'sig_fc_layer_{}'.format(block_index),
	)
	inter_h = model.FC(
		tanh_h, 
		sig_n,
		weight_optim=weight_optim,
		bias_optim=bias_optim,
		name = 'inter_embed_layer_{}'.format(block_index),
	)
	sig_h = model.Add(
		[inter_h, sig_h],
		'inter_add_layer_{}'.format(block_index),
	)
	if not linear_activation:
		sig_h = model.Sigmoid(
			[sig_h],
			'sig_tranfer_layer_{}'.format(block_index),
		)
		tanh_h = model.Tanh(
			[tanh_h], 
			'tanh_tranfer_layer_{}'.format(block_index),
		)
	return sig_h, tanh_h

def build_pinn(
	model,
	sig_net_dim=[1], tanh_net_dim=[1],
	weight_optim=None,
	bias_optim=None,
	loss_function='scaled_l1',
	max_loss_scale=1.0,
):
	'''
		sig_net_dim and tanh_net_dim are the lists of dimensions for each hidden
		layers in the sig_net and tanh_net respectively.
	'''
	assert len(sig_net_dim) * len(tanh_net_dim) > 0, 'arch cannot be empty'
	assert len(sig_net_dim) == len(tanh_net_dim), 'arch mismatch'
	assert sig_net_dim[-1] == tanh_net_dim[-1], 'last dim mismatch'

	block_index = 0
	sig_h = model.input_feature_schema.sig_input
	tanh_h = model.input_feature_schema.tanh_input
	for sig_n, tanh_n in zip(
		sig_net_dim, tanh_net_dim
	):
		sig_h, tanh_h = build_block(
			model,
			sig_h, tanh_h,
			sig_n, tanh_n,
			block_index,
			weight_optim=weight_optim,
			bias_optim=bias_optim,
		)
		block_index += 1

	pred = model.Mul([sig_h, tanh_h], model.trainer_extra_schema.prediction)
	# Add loss
	assert max_loss_scale > 1, 'max loss scale must > 1'

	loss_and_metrics = model.BatchDirectWeightedL1Loss(
		model.trainer_extra_schema,
		max_scale=max_loss_scale,
	)
	# Add metric
	model.add_metric_field('l1_metric', loss_and_metrics.l1_metric)
	model.add_metric_field('scaled_l1_metric', loss_and_metrics.scaled_l1_metric)

	if loss_function == 'scaled_l2':
		print('[Pi-NN Build Net]: Use scaled_l2 loss, but l1 metrics.')
		loss_and_metrics = model.BatchDirectWeightedL2Loss(
			model.trainer_extra_schema,
			max_scale=max_loss_scale,
		)

	model.add_loss(loss_and_metrics.loss)
	# Set output
	model.output_schema.pred.set_value(pred.get(), unsafe=True)
	model.output_schema.loss.set_value(loss_and_metrics.loss.get(), unsafe=True)

	return pred, loss_and_metrics.loss

def init_model_with_schemas(
	model_name, 
	sig_input_dim, tanh_input_dim,
	pred_dim
):
	workspace.ResetWorkspace()
	input_record_schema = schema.Struct(
		('sig_input', schema.Scalar((np.float32, (sig_input_dim, )))),
		('tanh_input', schema.Scalar((np.float32, (tanh_input_dim, ))))
	)
	output_record_schema = schema.Struct(
		('loss', schema.Scalar((np.float32, (1, )))),
		('pred', schema.Scalar((np.float32, (pred_dim, ))))
	)
	# use trainer_extra_schema as the loss input record
	trainer_extra_schema = schema.Struct(
		('label', schema.Scalar((np.float32, (pred_dim, )))),
		('prediction', schema.Scalar((np.float32, (pred_dim, ))))
	)
	model = layer_model_helper.LayerModelHelper(
		model_name,
		input_record_schema,
		trainer_extra_schema
	)
	model.output_schema = output_record_schema
	return model


if __name__ == '__main__':
	pass
	 

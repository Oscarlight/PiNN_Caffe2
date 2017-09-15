import caffe2_paths

from caffe2.python import (
	schema, optimizer, net_drawer, workspace, layer_model_helper
)
import numpy as np

def build_block(
	model,
	sig_input, tanh_input,
	sig_n, tanh_n, embed_n,
	block_index,
	optim=None,
	tranfer_before_interconnect=False,
	interconnect_method='Add',
	linear_activation = False
):
	tanh_h = model.FCWithoutBias(
		tanh_input, 
		tanh_n,
		weight_optim=optim,
		name = 'tanh_fc_layer_{}'.format(block_index),
	)
	if not linear_activation and tranfer_before_interconnect:
		tanh_h = model.Tanh(
			[tanh_h],
			name = 'tanh_tranfer_layer_{}'.format(block_index),
		)
	if embed_n > 0:
		inter_h = model.FCWithoutBias(
			tanh_h, 
			embed_n,
			weight_optim=optim,
			name = 'inter_embed_layer_{}'.format(block_index),
		)
		if interconnect_method == 'Concat':
			sig_input = model.Concat(
				[inter_h, sig_input],
				'inter_concat_layer_{}'.format(block_index),
				axis = 1
			)
		elif interconnect_method == 'Add':
			sig_input = model.Add(
				[inter_h, sig_input],
				'inter_add_layer_{}'.format(block_index),
			)
		else:
			raise Exception('Interconnect method: {} is not implemented.'.format(
					interconnect_method
				)
			)

	sig_h = model.FC(
		sig_input,
		sig_n,
		weight_optim=optim,
		bias_optim=optim,
		name = 'sig_fc_layer_{}'.format(block_index),
	)
	
	if not linear_activation:
		sig_h = model.Sigmoid(
			[sig_h],
			'sig_tranfer_layer_{}'.format(block_index),
		)
		if not tranfer_before_interconnect:
			tanh_h = model.Tanh(
				[tanh_h], 
				'tanh_tranfer_layer_{}'.format(block_index),
			)
	return sig_h, tanh_h

def build_pinn(
	model,
	label,
	sig_net_dim=[1], tanh_net_dim=[1], inner_embed_dim=[0],
	optim=None,
	tranfer_before_interconnect=False,
	interconnect_method='Add' 
):
	'''
		sig_net_dim and tanh_net_dim are the lists of dimensions for each hidden
		layers in the sig_net and tanh_net respectively.

		Precondition: when using Add as the interconncet method, the inner_embed_dim has 
		to be the same as the dimension of the last sig_net layer.
	'''
	assert len(sig_net_dim) * len(tanh_net_dim) > 0, 'arch cannot be empty'
	assert len(sig_net_dim) == len(tanh_net_dim), 'arch mismatch'
	assert sig_net_dim[-1] == tanh_net_dim[-1], 'last dim mismatch'

	block_index = 0
	sig_h, tanh_h = build_block(
		model,
		model.input_feature_schema.input_1,
		model.input_feature_schema.input_2,
		sig_net_dim[0], tanh_net_dim[0], inner_embed_dim[0],
		block_index,
		optim=optim,
		tranfer_before_interconnect = tranfer_before_interconnect,
		interconnect_method = interconnect_method,
	)

	for sig_n, tanh_n, embed_n in zip(
		sig_net_dim[1:], tanh_net_dim[1:], inner_embed_dim[1:]
	):
		block_index += 1
		# Use linear activation function in the last layer for the regression 
		linear_activation = True if block_index == len(sig_net_dim) else False
		sig_h, tanh_h = build_block(
			model,
			sig_h, tanh_h,
			sig_n, tanh_n, embed_n,
			block_index,
			optim=optim,
			tranfer_before_interconnect = tranfer_before_interconnect,
			interconnect_method = interconnect_method,
			linear_activation = linear_activation,
		)

	pred = model.Mul([sig_h, tanh_h], model.trainer_extra_schema.prediction)
	# Add loss
	loss = model.BatchDirectMSELoss(model.trainer_extra_schema)
	model.add_loss(loss)
	# Set output
	model.output_schema.pred.set_value(pred.get(), unsafe=True)
	model.output_schema.loss.set_value(loss.get(), unsafe=True)

	return pred, loss

def init_model_with_schemas(
	model_name, 
	sig_input_dim, tanh_input_dim,
	pred_dim
):
	workspace.ResetWorkspace()
	input_record_schema = schema.Struct(
		('input_1', schema.Scalar((np.float32, (sig_input_dim, )))), # sig
		('input_2', schema.Scalar((np.float32, (tanh_input_dim, ))))  # tanh
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
	 

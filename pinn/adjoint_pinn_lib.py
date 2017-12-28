import caffe2_paths

from caffe2.python import (
	core, workspace, layer_model_helper, schema, optimizer, scope
)
from caffe2.python.layers.tags import Tags
from caffe2.python.modeling.parameter_sharing import (
    ParameterSharing,
)
import numpy as np

class TrainTarget(object):
	ORIGIN = 'origin'
	ADJOINT = 'adjoint'

def build_origin_block(
	model,
	sig_input, tanh_input,
	sig_n, tanh_n,
	block_index,
	weight_optim=None,
	bias_optim=None,
):
	tanh_h = model.FCWithoutBias(
		tanh_input, 
		tanh_n,
		weight_optim=weight_optim,
		name='tanh_fc_layer_{}'.format(block_index),
	)
	sig_h = model.FC(
		sig_input,
		sig_n,
		weight_optim=weight_optim,
		bias_optim=bias_optim,
		name='sig_fc_layer_{}'.format(block_index),
	)
	inter_h = model.FC(
		tanh_h, 
		sig_n,
		weight_optim=weight_optim,
		bias_optim=bias_optim,
		name='inter_embed_layer_{}'.format(block_index),
	)
	sig_h = model.Add(
		[inter_h, sig_h],
		'inter_add_layer_{}'.format(block_index),
	)
	sig_h = model.Sigmoid(
		[sig_h],
		'sig_tranfer_layer_{}'.format(block_index),
	)
	tanh_h = model.Tanh(
		[tanh_h], 
		'tanh_tranfer_layer_{}'.format(block_index),
	)
	return sig_h, tanh_h

def build_adjoint_block(
	model,
	beta, alpha, sig_h, tanh_h,
	sig_n, tanh_n,
	block_index,
	weight_optim=None,
):
	# sig_net_adjoint
	delta_ad = model.FCTransposeW(
		beta,
		sig_n,
		weight_optim=weight_optim,
		name='sig_fc_layer_{}'.format(block_index),
	)
	sig_ones = model.ConstantFill(
		[sig_h],
		'sig_ones_{}'.format(block_index),
		value=1.0, 
		dtype=core.DataType.FLOAT
	)
	sig_multiplier = model.Mul(
		[sig_h, model.Sub([sig_ones, sig_h], 'sig_sub_{}'.format(block_index))],
		'sig_multiplier_{}'.format(block_index),
	)
	beta = model.Mul(
		[delta_ad, sig_multiplier],
		'sig_output_beta_{}'.format(block_index)
	)
	# tanh_net_adjoint
	gamma_ad = model.FCTransposeW(
		alpha,
		tanh_n,
		weight_optim=weight_optim,
		name='tanh_fc_layer_{}'.format(block_index),
	)
	tanh_ones = model.ConstantFill(
		[tanh_h],
		'tanh_ones_{}'.format(block_index),
		value=1.0, 
		dtype=core.DataType.FLOAT
	)
	tanh_multiplier = model.Sub(
		[tanh_ones, model.Mul([tanh_h, tanh_h], 'tanh_mul_{}'.format(block_index))],
		'tanh_multiplier_{}'.format(block_index),
	)
	alpha = model.Mul(
		[gamma_ad, tanh_multiplier],
		'tanh_adjoint_layer_{}'.format(block_index)
	)
	inter = model.FCTransposeW(
		beta,
		tanh_n,
		weight_optim=weight_optim,
		name='inter_embed_layer_{}'.format(block_index-1),		
	)
	alpha = model.Add(
		[alpha, inter],
		'tanh_output_alpha_{}'.format(block_index)
	)
	return beta, alpha

def build_adjoint_pinn(
	model,
	sig_input_dim=1, tanh_input_dim=1,
	sig_net_dim=[1], tanh_net_dim=[1],
	weight_optim=None,
	bias_optim=None,
	adjoint_tag=Tags.EXCLUDE_FROM_PREDICTION,
	train_target=TrainTarget.ADJOINT,
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

	with ParameterSharing({'origin' : 'adjoint'}):
		sig_h_lst = []
		tanh_h_lst = []
		block_index = 0
		with scope.NameScope('origin'):
			sig_h = model.input_feature_schema.sig_input
			tanh_h = model.input_feature_schema.tanh_input
			for sig_n, tanh_n in zip(
				sig_net_dim, tanh_net_dim
			):
				sig_h, tanh_h = build_origin_block(
					model,
					sig_h, tanh_h,
					sig_n, tanh_n,
					block_index,
					weight_optim=weight_optim,
					bias_optim=bias_optim,
				)
				sig_h_lst.append(sig_h)
				tanh_h_lst.append(tanh_h)
				block_index += 1
			origin_pred = model.Mul(
				[sig_h, tanh_h], 
				'origin_pred'
			)
		with scope.NameScope('adjoint'):
			# adjoint_tag decides how we are going to use the adjoint net.
			with Tags(adjoint_tag):
				ad_input = model.input_feature_schema.adjoint_input
				sig_h = sig_h_lst[block_index-1]
				tanh_h = tanh_h_lst[block_index-1]
				# for the output, sig_h and tanh_h has the same dimention.
				output_ones = model.ConstantFill(
					[sig_h],
					'output_ones_{}'.format(block_index),
					value=1.0, 
					dtype=core.DataType.FLOAT
				)
				beta = model.Mul(
					[tanh_h, model.Mul(
						[sig_h, model.Sub([output_ones, sig_h], 
							'sig_output_sub_{}'.format(block_index)
						)], 'sig_output_mul_{}'.format(block_index)
					)], 'sig_output_beta_{}'.format(block_index)
				)
				alpha = model.Mul(
					[sig_h, model.Sub(
						[output_ones, model.Mul([tanh_h, tanh_h],
							'tanh_output_sq_{}'.format(block_index)
						)], 'tanh_output_sub_{}'.format(block_index)
					)], 'tanh_output_mul_{}'.format(block_index)
				)
				inter = model.FCTransposeW(
					beta,
					tanh_net_dim[-1],
					weight_optim=weight_optim,
					name='inter_embed_layer_{}'.format(block_index-1)					
				)
				alpha = model.Add(
					[alpha, inter],
					'tanh_output_alpha_{}'.format(block_index)
				)
				for sig_n, tanh_n in zip(
					reversed(sig_net_dim[:-1]), 
					reversed(tanh_net_dim[:-1])
				):
					block_index -= 1
					sig_h = sig_h_lst[block_index-1]
					tanh_h = tanh_h_lst[block_index-1]
					beta, alpha = build_adjoint_block(
						model,
						beta, alpha, sig_h, tanh_h,
						sig_n, tanh_n,
						block_index,
						weight_optim=weight_optim,
					)
				sig_adjoint_pred = model.FCTransposeW(
					beta,
					sig_input_dim,
					weight_optim=weight_optim,
					name='sig_fc_layer_{}'.format(block_index-1)
				)
				tanh_adjoint_pred = model.FCTransposeW(
					alpha,
					tanh_input_dim,
					weight_optim=weight_optim,
					name='tanh_fc_layer_{}'.format(block_index-1)
				)

		# Add loss
		if train_target == TrainTarget.ADJOINT:
			model.trainer_extra_schema.sig_loss_record.prediction.set_value(
				sig_adjoint_pred.get(), unsafe=True)
			model.trainer_extra_schema.tanh_loss_record.prediction.set_value(
				tanh_adjoint_pred.get(), unsafe=True)		
			# CAUTIONS: BatchDirectMSELoss calls SquaredL2Distance op, which assume 
			# the input are 1D vector
			sig_loss = model.BatchDirectMSELoss(
				model.trainer_extra_schema.sig_loss_record)
			tanh_loss = model.BatchDirectMSELoss(
				model.trainer_extra_schema.tanh_loss_record)
			adjoint_loss = model.Add([sig_loss, tanh_loss], 'adjoint_loss')
			model.add_loss(sig_loss)
			model.add_loss(tanh_loss)
			# Set output
			model.output_schema.sig_adjoint_pred.set_value(
				sig_adjoint_pred.get(), unsafe=True)
			model.output_schema.tanh_adjoint_pred.set_value(
				tanh_adjoint_pred.get(), unsafe=True)
			loss = adjoint_loss
		if train_target == TrainTarget.ORIGIN:
			model.trainer_extra_schema.origin_loss_record.prediction.set_value(
				origin_pred.get(), unsafe=True)
			# Add L1 Loss
			assert max_loss_scale > 1, 'max loss scale must > 1'
			loss_and_metrics = model.BatchDirectWeightedL1Loss(
				model.trainer_extra_schema.origin_loss_record,
				max_scale=max_loss_scale,
			)
			# Add metric
			model.add_metric_field('l1_metric', 
				loss_and_metrics.l1_metric)
			model.add_metric_field('scaled_l1_metric', 
				loss_and_metrics.scaled_l1_metric)
			if loss_function == 'scaled_l2':
				print('[Pi-NN Build Net]: Use scaled_l2 loss, but l1 metrics.')
				loss_and_metrics = model.BatchDirectWeightedL2Loss(
					model.trainer_extra_schema.origin_loss_record,
					max_scale=max_loss_scale,
				)
			model.add_loss(loss_and_metrics.loss)
			loss = loss_and_metrics.loss
		else:
			raise Exception('train target: ' + train_target + ' not implemented')

		model.output_schema.origin_pred.set_value(
			origin_pred.get(), unsafe=True)
		model.output_schema.loss.set_value(
			loss.get(), unsafe=True)

		return origin_pred, sig_adjoint_pred, tanh_adjoint_pred, loss


def init_adjoint_model_with_schemas(
	model_name, 
	sig_input_dim, tanh_input_dim,
	pred_dim,
	train_target=TrainTarget.ADJOINT
):
	'''
	 output_records have to filled with existing blobs.
	'''
	workspace.ResetWorkspace()
	print('>>> Training Target: ' + train_target)
	if train_target == TrainTarget.ADJOINT:
		# When training the adjoint network, we also need to forward pass
		# through the origin network
		input_record_schema = schema.Struct(
			('sig_input', schema.Scalar((np.float32, (sig_input_dim, )))), # sig
			('tanh_input', schema.Scalar((np.float32, (tanh_input_dim, )))),  # tanh
			('adjoint_input', schema.Scalar((np.float32, (pred_dim, ))))
		)
		output_record_schema = schema.Struct(
			('loss', schema.Scalar((np.float32, (1, )))),
			('origin_pred', schema.Scalar((np.float32, (pred_dim, )))),
			('sig_adjoint_pred', schema.Scalar((np.float32, (sig_input_dim, )))),
			('tanh_adjoint_pred', schema.Scalar((np.float32, (tanh_input_dim, )))),
		)
		# use trainer_extra_schema as the loss input record
		trainer_extra_schema = schema.Struct(
			('sig_loss_record', schema.Struct(
				('label', schema.Scalar((np.float32, 
					(sig_input_dim, )))),
				('prediction', schema.Scalar((np.float32, 
					(sig_input_dim, ))))
				)
			),
			('tanh_loss_record', schema.Struct(
				('label', schema.Scalar((np.float32, 
					(tanh_input_dim, )))),
				('prediction', schema.Scalar((np.float32, 
					(tanh_input_dim, ))))
				)
			),
			('origin_loss_record', schema.Struct(
				('label', schema.Scalar((np.float32, 
					(pred_dim, )))),
				('prediction', schema.Scalar((np.float32, 
					(pred_dim, ))))
				)
			),
		)
	if train_target == TrainTarget.ORIGIN:
		# When training the origin network, no need of the adjoint network
		input_record_schema = schema.Struct(
			('sig_input', schema.Scalar((np.float32, (sig_input_dim, )))), # sig
			('tanh_input', schema.Scalar((np.float32, (tanh_input_dim, )))),  # tanh
			('adjoint_input', schema.Scalar((np.float32, (pred_dim, ))))
		)
		output_record_schema = schema.Struct(
			('loss', schema.Scalar((np.float32, (1, )))),
			('origin_pred', schema.Scalar((np.float32, (pred_dim, )))),
		)
		# use trainer_extra_schema as the loss input record
		trainer_extra_schema = schema.Struct(
			('origin_loss_record', schema.Struct(
				('label', schema.Scalar((np.float32, 
					(pred_dim, )))),
				('prediction', schema.Scalar((np.float32, 
					(pred_dim, ))))
				)
			),
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
	 

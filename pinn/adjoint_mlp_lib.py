import caffe2_paths

from caffe2.python import (
	core, workspace, layer_model_helper, schema, optimizer, scope
)
from caffe2.python.modeling.parameter_sharing import (
    ParameterSharing,
)
from caffe2.python.layers.tags import Tags
import numpy as np

def build_adjoint_mlp(
	model,
	input_dim = 1,
	hidden_dims = [5, 5],
	output_dim = 1,
	optim=None,
):
	''' Precondition:
			model.input_feature_schema.origin_input has shape of (input_dim, )
			model.input_feature_schema.adjoint_input has shape of (output_dim, )
		Note:
			adjoint_input is binary array, e.g. [1, 0], which is used as the 
			"selecter".
	'''
	assert len(hidden_dims) >= 1, "at least one hidden dim"
	with ParameterSharing({'origin' : 'adjoint'}):
		z = model.input_feature_schema.origin_input 
		z_lst = []
		idx = 0
		with scope.NameScope('origin'):
			for hidden_dim in hidden_dims:
				gamma = model.FC(
					z,
					hidden_dim, 
					weight_optim=optim,
					bias_optim=optim,
					name='fc{}'.format(idx)
				)
				z = model.Sigmoid(gamma, 'sig{}'.format(idx))
				z_lst.append(z)
				idx += 1
			# Output layer: no grad for the bias in this layer,
			# use FCWithoutBias
			origin_pred = model.FCWithoutBias(
				z, 
				output_dim,
				weight_optim=optim, 
				name='fc{}'.format(idx)
			)
			origin_pred = model.NanCheck(origin_pred, 'origin_pred')

		with scope.NameScope('adjoint'):
			# with Tags(Tags.EXCLUDE_FROM_PREDICTION):
			alpha = model.input_feature_schema.adjoint_input
			for hidden_dim in reversed(hidden_dims):
				gamma_ad = model.FCTransposeW(
					alpha, 
					hidden_dim,
					weight_optim=optim,
					name='fc{}'.format(idx)
				)
				z = z_lst[idx-1]
				# Note: passing gradient is helpful
				# z = model.StopGradient(z, z)
				# TODO: use add_global_constant
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
				alpha = model.Mul(
					[gamma_ad, multiplier], 
					'adjoint_layer{}'.format(idx)
				)
				idx -= 1
			adjoint_pred = model.FCTransposeW(
				alpha, 
				input_dim,
				weight_optim=optim,
				name='fc{}'.format(idx)
			)
	# Add loss
	model.trainer_extra_schema.prediction.set_value(adjoint_pred.get(), unsafe=True)
	loss = model.BatchDirectMSELoss(model.trainer_extra_schema)
	model.add_loss(loss)
	# Set output
	model.output_schema.origin_pred.set_value(origin_pred.get(), unsafe=True)
	model.output_schema.adjoint_pred.set_value(adjoint_pred.get(), unsafe=True)
	model.output_schema.loss.set_value(loss.get(), unsafe=True)

	return origin_pred, adjoint_pred, loss

def init_model_with_schemas(
	model_name, 
	input_dim, output_dim,
):
	workspace.ResetWorkspace()
	input_record_schema = schema.Struct(
		('origin_input', schema.Scalar((np.float32, (input_dim, )))),
		('adjoint_input', schema.Scalar((np.float32, (output_dim, ))))
	)
	output_record_schema = schema.Struct(
		('loss', schema.Scalar((np.float32, (input_dim, )))),
		('origin_pred', schema.Scalar((np.float32, (output_dim, )))),
		('adjoint_pred', schema.Scalar((np.float32, (input_dim, ))))
	)
	# use trainer_extra_schema as the loss input record
	pred_dim = input_dim # for adjoint nn, the pred_dim is the same as original input
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
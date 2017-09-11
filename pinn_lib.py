import caffe2_paths

from caffe2.python import (
	schema, optimizer, net_drawer, workspace, layer_model_helper
)
import numpy as np

def build_block(
	model, 
	sig_input, tanh_input,
	sig_n, tanh_n, embed_n,
	optim=None,
	tranfer_before_interconnect=False,
	concat_embed=False
):
	tanh_h = model.FCWithoutBias(
		tanh_input, 
		tanh_n,
		weight_optim=optim,
		name = model.next_layer_name('tanh_fc_layer')
	)
	if tranfer_before_interconnect:
		tanh_h = model.Tanh(
			[tanh_h],
			name = model.next_layer_name('tanh_tranfer_layer')
		)
	if embed_n > 0:
		inter_h = model.FCWithoutBias(
			tanh_h, 
			embed_n,
			weight_optim=optim,
			name = model.next_layer_name('inter_embed_layer')
		)
		if concat_embed:
			sig_input = model.Concat(
				[inter_h, sig_input],
				model.next_layer_name('sig_concat_layer'),
				axis = 1
			)
		else:
			#TODO: assert dim is the same
			sig_input = model.Add(
				[inter_h, sig_input],
				model.next_layer_name('sig_add_layer')
			)

	sig_h = model.FC(
		sig_input,
		sig_n,
		weight_optim=optim,
		bias_optim=optim,
		name = model.next_layer_name('sig_fc_layer'),
	)
	sig_h = model.Sigmoid(
		[sig_h],
		model.next_layer_name('sig_tranfer_layer'),
	)
	if not tranfer_before_interconnect:
		tanh_h = model.Tanh(
			[tanh_h], 
			model.next_layer_name('tanh_tranfer_layer'),
		)
	return sig_h, tanh_h

def _build_pinn_impl(
	model, 
	sig_net_dim=[1], tanh_net_dim=[1], inner_embed_dim=[0],
	optim=None,
	tranfer_before_interconnect=False,
	concat_embed=False
):
	assert len(sig_net_dim) * len(tanh_net_dim) > 0, 'arch cannot be empty'
	assert len(sig_net_dim) == len(tanh_net_dim), 'arch mismatch'
	assert sig_net_dim[-1] == tanh_net_dim[-1], 'last dim mismatch'
	sig_h, tanh_h = build_block(
		model,
		model.input_feature_schema.sig_input,
		model.input_feature_schema.tanh_input,
		sig_net_dim[0], tanh_net_dim[0], inner_embed_dim[0],
		optim=optim,
		tranfer_before_interconnect = tranfer_before_interconnect,
		concat_embed = concat_embed,
	)
	for sig_n, tanh_n, embed_n in zip(
		sig_net_dim[1:], tanh_net_dim[1:], inner_embed_dim[1:]
	):
		sig_h, tanh_h = build_block(
			model,
			sig_h, tanh_h,
			sig_n, tanh_n, embed_n,
			optim=optim,
			tranfer_before_interconnect = tranfer_before_interconnect,
			concat_embed = concat_embed,
		)
	output = model.Mul([sig_h, tanh_h], 'prediction')
	return output

def build_pinn(
	model,
	label,
	num_label,
	sig_net_dim=[1], tanh_net_dim=[1], inner_embed_dim=[0],
	optim=None,
	tranfer_before_interconnect=False,
	concat_embed=False
):

	pred = _build_pinn_impl(
		model, 
		sig_net_dim=sig_net_dim, 
		tanh_net_dim=tanh_net_dim, 
		inner_embed_dim=inner_embed_dim,
		optim=optim,
		tranfer_before_interconnect=tranfer_before_interconnect,
		concat_embed=concat_embed
	)
	model.trainer_extra_schema.prediction.set_value(pred.get(), unsafe=True)
	loss = model.BatchDirectMSELoss(model.trainer_extra_schema)
	model.output_schema.pred.set_value(pred.get(), unsafe=True)
	model.output_schema.loss.set_value(loss.get(), unsafe=True)
	model.add_loss(loss)
	return pred, loss

def init_model_with_schemas(
	model_name, 
	sig_input_dim, tanh_input_dim,
	pred_dim
):
	workspace.ResetWorkspace()
	input_record_schema = schema.Struct(
		('sig_input', schema.Scalar((np.float32, (tanh_input_dim, )))),
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
	 

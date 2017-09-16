import caffe2_paths
import os
from caffe2.python import (
	workspace, layer_model_helper, schema, optimizer, net_drawer
)
import caffe2.python.layer_model_instantiator as instantiator
import numpy as np
from pinn_lib import build_pinn, init_model_with_schemas
import data_reader
import preproc
import parser

class DCModel:
	def __init__(
		self, 
		model_name, 
		train_file_name=None,
		eval_file_name=None,
		preproc_slope=0,
		preproc_threshold=0,
	):
		self.model = init_model_with_schemas(model_name, 1, 1, 1)
		self.input_data_store = {}
		if train_file_name:
			self.add_data(
				'train', file_name=train_file_name, 
				preproc_slope=preproc_slope,
				preproc_threshold=preproc_threshold
			)
		if eval_file_name:
			self.add_data(
				'eval', file_name=eval_file_name,
				preproc_slope=preproc_slope,
				preproc_threshold=preproc_threshold
			)
		self.net_store = {}

	def build_nets(
		self,
		hidden_sig_dims, 
		hidden_tanh_dims,
		train_batch_size=1,
		test_batch_size=1,
		transfer_before_interconnect=False,
		interconnect_method='Add',
		inner_embed_dims=[],
		optim_param = {'alpha':0.01, 'epsilon':1e-4} 
	):
		assert len(self.input_data_store) > 0, 'Input data store is empty.'
		assert 'train' in input_data_store, 'Missing training data.'
		# Build the date reader net for train net
		input_data_train = data_reader.build_input_reader(
			self.model, 
			input_data_store['train'][0], 
			'minidb', 
			['sig_input', 'tanh_input'], 
			batch_size=train_batch_size,
			data_type='train',
		)
		if 'eval' in input_data_store:
			# Build the data reader net for eval net
			input_data_eval = data_reader.build_input_reader(
				self.model, 
				input_data_store['eval'][0], 
				'minidb', 
				['sig_input', 'tanh_input'], 
				batch_size=eval_batch_size,
				data_type='eval',
			)

		# Build the computational net
		if interconnect_method == 'Add':
			inner_embed_dims = [1] # sig_net input dim
			for sig_dim in hidden_sig_dims[:-1]:
				inner_embed_dims.append(sig_dim)
		elif interconnect_method == 'Concat':
			assert len(inner_embed_dims) == len(hidden_sig_dims), 'invalid inner_embed_dims'

		pred, loss = build_pinn(
			self.model,
			label,
			sig_net_dim = hidden_sig_dims,
			tanh_net_dim = hidden_tanh_dims,
			inner_embed_dim = inner_embed_dims,
			optim=optimizer.AdagradOptimizer(**optim_param),
			tranfer_before_interconnect=transfer_before_interconnect, 
			interconnect_method=interconnect_method
		)

		# Create train net
		model.input_feature_schema.sig_input.set_value(
			input_data_train[0].get(), unsafe=True)
		model.input_feature_schema.tanh_input.set_value(
			input_data_train[1].get(), unsafe=True)
		model.trainer_extra_schema.label.set_value(
			input_data_train[2].get(), unsafe=True)

		train_init_net, train_net = instantiator.generate_training_nets(self.model)
		workspace.RunNetOnce(train_init_net)
		workspace.CreateNet(train_net)
		self.net_store['train_net'] = train_net

		if 'eval' in input_data_store:
			# Create eval net
			model.input_feature_schema.sig_input.set_value(
				input_data_eval[0].get(), unsafe=True)
			model.input_feature_schema.tanh_input.set_value(
				input_data_eval[1].get(), unsafe=True)
			model.trainer_extra_schema.label.set_value(
				input_data_eval[2].get(), unsafe=True)

			eval_net = instantiator.generate_eval_net(model)
			workspace.CreateNet(eval_net)
			self.net_store['eval_net'] = eval_net


	def add_data(
		self,
		data_tag,
		data_arrays=[], 
		file_name=None, 
		preproc_slope=0, 
		preproc_threshold=0
	):
		if len(data_arrays) > 0:
			print('Read into data directly from data arrays... ' + 
				'Note: data are in the order of vg, vd, and id')
		elif file_name:
			print('Read in the data from data file...')
			file_name_wo_ext, file_ext = os.path.splitext(file_name)
			if file_ext == '.mdm':
				data_arrays = parser.read_dc_iv_mdm(file_name)
			elif file_ext == '.csv':
				data_arrays = parser.read_dc_iv_csv(file_name)
			else:
				raise Exception('Unsupported data format.')
		else:
			raise Exception('No data source.')

		db_name = file_name_wo_ext + '.minidb'
		restore_func = None
		if not os.path.isfile(db_name):
			print(">>> Create a new database...")	
			# preproc the data
			assert len(data_arrays) == 3, 'Incorrect number of input data'
			scale, vg_shift = preproc.compute_dc_meta(*data_arrays)
			preproc_data_arrays, restore_func = preproc.dc_iv_preproc(
				data_arrays[0], data_arrays[1], data_arrays[2], 
				scale, vg_shift, 
				slope=preproc_slope,
				threshold=preproc_threshold
			)
			preproc_data_arrays = [np.expand_dims(
				x, axis=1) for x in preproc_data_arrays]
			data_reader.write_db('minidb', db_name, *preproc_data_arrays)
		else:
			print(">>> The database with the same name already existed.")

		self.input_data_store[data_tag] = (db_name, restore_func)

	
		
		 
			
			
			
			

import caffe2_paths
import os
import pickle
from caffe2.python import (
	workspace, layer_model_helper, schema, optimizer, net_drawer
)
import caffe2.python.layer_model_instantiator as instantiator
import numpy as np
from pinn_lib import build_pinn, init_model_with_schemas
import data_reader
import preproc
import parser
# import logging
import matplotlib.pyplot as plt

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
		self.preproc_param = {}
		self.net_store = {}
		self.data_arrays_dict = {}
		self.reports = {'epoch':[],'train_loss':[], 'eval_loss':[]}
		self.file_names = {}
		if train_file_name:
			self.add_data(
				'train', file_name=train_file_name, 
				preproc_slope=preproc_slope,
				preproc_threshold=preproc_threshold
			)
		if eval_file_name:
			self.add_data(
				'eval', file_name=eval_file_name,
			)

	def build_nets(
		self,
		hidden_sig_dims, 
		hidden_tanh_dims,
		batch_size=1,
		transfer_before_interconnect=False,
		interconnect_method='Add',
		inner_embed_dims=[],
		optim_param = {'alpha':0.01, 'epsilon':1e-4} 
	):
		assert len(self.input_data_store) > 0, 'Input data store is empty.'
		assert 'train' in self.input_data_store, 'Missing training data.'
		self.batch_size = batch_size
		# Build the date reader net for train net
		input_data_train = data_reader.build_input_reader(
			self.model, 
			self.input_data_store['train'][0], 
			'minidb', 
			['sig_input', 'tanh_input'], 
			batch_size=batch_size,
			data_type='train',
		)

		if 'eval' in self.input_data_store:
			# Build the data reader net for eval net
			input_data_eval = data_reader.build_input_reader(
				self.model, 
				self.input_data_store['eval'][0], 
				'minidb', 
				['sig_input', 'tanh_input'], 
				batch_size=batch_size,
				data_type='eval',
			)

		# Build the computational net
		if interconnect_method == 'Add':
			inner_embed_dims = [1] # sig_net input dim
			for sig_dim in hidden_sig_dims[:-1]:
				inner_embed_dims.append(sig_dim)
		elif interconnect_method == 'Concat':
			assert len(inner_embed_dims) == len(hidden_sig_dims), 'invalid inner_embed_dims'
		else:
			raise Exception('Unsupported data format.')

		# Create train net
		self.model.input_feature_schema.sig_input.set_value(
			input_data_train[0].get(), unsafe=True)
		self.model.input_feature_schema.tanh_input.set_value(
			input_data_train[1].get(), unsafe=True)
		self.model.trainer_extra_schema.label.set_value(
			input_data_train[2].get(), unsafe=True)

		self.pred, self.loss = build_pinn(
			self.model,
			sig_net_dim = hidden_sig_dims,
			tanh_net_dim = hidden_tanh_dims,
			inner_embed_dim = inner_embed_dims,
			optim=optimizer.AdagradOptimizer(**optim_param),
			tranfer_before_interconnect=transfer_before_interconnect, 
			interconnect_method=interconnect_method
		)

		train_init_net, train_net = instantiator.generate_training_nets(self.model)
		workspace.RunNetOnce(train_init_net)
		workspace.CreateNet(train_net)
		self.net_store['train_net'] = train_net

		if 'eval' in self.input_data_store:
			# Create eval net
			self.model.input_feature_schema.sig_input.set_value(
				input_data_eval[0].get(), unsafe=True)
			self.model.input_feature_schema.tanh_input.set_value(
				input_data_eval[1].get(), unsafe=True)
			self.model.trainer_extra_schema.label.set_value(
				input_data_eval[2].get(), unsafe=True)
			eval_net = instantiator.generate_eval_net(self.model)
			workspace.CreateNet(eval_net)
			self.net_store['eval_net'] = eval_net

		pred_net = instantiator.generate_predict_net(self.model)
		workspace.CreateNet(pred_net)
		self.net_store['pred_net'] = pred_net

	def add_data(
		self,
		data_tag,
		data_arrays=[], 
		file_name=None, 
		preproc_slope=0, 
		preproc_threshold=0,
		save_data_arrays=False,
	):
		if len(data_arrays) > 0:
			print('>>> Read into data directly from data arrays... ' + 
				'Note: data are in the order of vg, vd, and id')
		elif file_name:
			print('>>> Read in the data from data file : {}...'.format(file_name))
			file_name_wo_ext, file_ext = os.path.splitext(file_name)
			if file_ext == '.mdm':
				data_arrays = parser.read_dc_iv_mdm(file_name)
			elif file_ext == '.csv':
				data_arrays = parser.read_dc_iv_csv(file_name)
			else:
				raise Exception('Unsupported data format.')
		else:
			raise Exception('No data source.')

		# number of examples
		num_example = len(data_arrays[0])
		for data in data_arrays[1:]:
			assert len(data) == num_example, 'Mismatch dimensions'
		self.file_names[data_tag] = file_name_wo_ext
		db_name = file_name_wo_ext + '.minidb'
		if not os.path.isfile(db_name):
			print("+++ Create a new database...")	
			# preproc the data
			assert len(data_arrays) == 3, 'Incorrect number of input data'
			if len(self.preproc_param) == 0:
				# Compute the meta data if not set yet
				scale, vg_shift = preproc.compute_dc_meta(*data_arrays)
				self.preproc_param = {
					'scale' : scale, 
					'vg_shift' : vg_shift, 
					'preproc_slope' : preproc_slope, 
					'preproc_threshold' : preproc_threshold
				}
				pickle.dump(
					self.preproc_param, 
					open(file_name_wo_ext + '.p', 'wb')
				)
			preproc_data_arrays = preproc.dc_iv_preproc(
				data_arrays[0], data_arrays[1], data_arrays[2], 
				self.preproc_param['scale'], 
				self.preproc_param['vg_shift'], 
				slope=self.preproc_param['preproc_slope'],
				threshold=self.preproc_param['preproc_threshold']
			)
			if save_data_arrays:
				self.data_arrays_dict[data_tag] = data_arrays
			preproc_data_arrays = [np.expand_dims(
				x, axis=1) for x in preproc_data_arrays]
			data_reader.write_db('minidb', db_name, *preproc_data_arrays)
		else:
			print("--- The database with the same name already existed.")

		self.input_data_store[data_tag] = [db_name, num_example]


	def train_with_eval(
		self,
		num_epoch=1,
		report_interval=0,
		eval_during_training=False,
	):
		''' Speed Comparison

		'''
		num_batch_per_epoch = int(
			self.input_data_store['train'][1] / 
			self.batch_size
		)
		if not self.input_data_store['train'][1] % self.batch_size == 0:
			num_batch_per_epoch += 1
			print('[Warning]: batch_size cannot be divided. ' + 
				'Run on {} example instead of {}'.format(
						num_batch_per_epoch * self.batch_size,
						self.input_data_store['train'][1]
					)
				)
		print('<<< Run {} iteration'.format(num_epoch * num_batch_per_epoch))

		train_net = self.net_store['train_net']
		if report_interval > 0:
			print('>>> Training with Reports')
			num_eval = int(num_epoch / report_interval)
			num_unit_iter = int((num_batch_per_epoch * num_epoch)/num_eval)
			if eval_during_training and 'eval_net' in self.net_store:
				print('>>> Training with Eval Reports (Slowest mode)')
				eval_net = self.net_store['eval_net']
			for i in range(num_eval):
				workspace.RunNet(
					train_net.Proto().name, 
					num_iter=num_unit_iter
				)
				self.reports['epoch'].append((i + 1) * report_interval)
				train_loss = np.asscalar(schema.FetchRecord(self.loss).get())
				self.reports['train_loss'].append(train_loss)
				if eval_during_training and 'eval_net' in self.net_store:
					workspace.RunNet(
						eval_net.Proto().name,
						num_iter=num_unit_iter)
					eval_loss = np.asscalar(schema.FetchRecord(self.loss).get())
					self.reports['eval_loss'].append(eval_loss)
		else:
			print('>>> Training without Reports (Fastest mode)')
			workspace.RunNet(
				train_net, 
				num_iter=num_epoch * num_batch_per_epoch
			)


	def avg_loss_full_epoch(self, net_name):
		num_batch_per_epoch = int(
			self.input_data_store['train'][1] / 
			self.batch_size
		)
		if not self.input_data_store['train'][1] % self.batch_size == 0:
			num_batch_per_epoch += 1
			print('[Warning]: batch_size cannot be divided. ' + 
				'Run on {} example instead of {}'.format(
						num_batch_per_epoch * self.batch_size,
						self.input_data_store['train'][1]
					)
				)
		# Get the average loss of all data
		loss = 0.
		for j in range(num_batch_per_epoch):
			workspace.RunNet(self.net_store[net_name])
			loss += np.asscalar(schema.FetchRecord(self.loss).get())
		loss /= num_batch_per_epoch
		return loss

	def draw_nets(self):
		for net_name in self.net_store:
			net = self.net_store[net_name]
			graph = net_drawer.GetPydotGraph(net.Proto().op, rankdir='TB')
			with open(net.Name() + ".png",'wb') as f:
				f.write(graph.create_png())

	def get_input_data_restore_func(self, pickle_file_name):
		self.preproc_param = favorite_color = pickle.load(
			open(pickle_file_name, "rb" )
		)
		return get_restore_func(
				self.preproc_param['scale'], 
				self.preproc_param['vg_shift'], 
				slope=self.preproc_param['preproc_slope'],
				threshold=self.preproc_param['preproc_threshold']
			)

	def plot_IV(self):
		'''
		eval results
		'''
		eval_net = self.net_store['eval_net']
		workspace.RunNet(eval_net)
		vg = schema.FetchRecord(self.model.input_feature_schema.sig_input).get()
		vd = schema.FetchRecord(self.model.input_feature_schema.tanh_input).get()
		ids = schema.FetchRecord(self.pred).get()

	def plot_loss_trend(self):
		plt.plot(self.reports['epoch'], self.reports['train_loss'])
		if len(self.reports['eval_loss']) > 0:
			plt.plot(self.reports['epoch'], self.reports['eval_loss'], 'r--')
		plt.show()


		 
			
			
			
			

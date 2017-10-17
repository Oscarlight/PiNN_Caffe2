import caffe2_paths
import os
import pickle
from caffe2.python import (
	workspace, layer_model_helper, schema, optimizer, net_drawer
)
import caffe2.python.layer_model_instantiator as instantiator
import numpy as np
from pinn.adjoint_mlp_lib import build_adjoint_mlp, init_model_with_schemas
import pinn.data_reader as data_reader
import pinn.preproc as preproc
import pinn.parser as parser
import pinn.visualizer as visualizer
import pinn.exporter as exporter
# import logging
import matplotlib.pyplot as plt

class ACQVModel:
	def __init__(
		self, 
		model_name,
		input_dim=1,
		output_dim=1,
	):	
		self.model_name = model_name
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.model = init_model_with_schemas(
			model_name, self.input_dim, self.output_dim)
		self.input_data_store = {}
		self.preproc_param = {}
		self.net_store = {}
		self.reports = {'epoch':[],'train_loss':[], 'eval_loss':[]}


	def add_data(
		self,
		data_tag,
		data_arrays, 
		preproc_param,
		override=True,
	):
		'''
		data_arrays are in the order of origin_input, adjoint_label
		origin_input and adjoint_label must be numpy arrays
		'''
		#check length and dimensions of origin input and adjoint label 
		assert len(data_arrays) == 2, 'Incorrect number of input data'
		voltages = data_arrays[0]
		capas = data_arrays[1]
		assert voltages.shape == capas.shape, 'Mismatch dimensions'
		
		#Set preprocess parameters and database name
		self.preproc_param = preproc_param
		self.pickle_file_name = self.model_name + '_preproc_param' + '.p'
		db_name = self.model_name + '_' + data_tag + '.minidb'

		if os.path.isfile(db_name):
			if override:
				print("XXX Delete the old database...")
				os.remove(db_name)
				os.remove(self.pickle_file_name)
			else:
				raise Exception('Encounter database with the same name. ' +
					'Choose the other model name or set override to True.')
		print("+++ Create a new database...")	
		
		self.preproc_param.setdefault('max_loss_scale', 1.)
		
		pickle.dump(
			self.preproc_param, 
			open(self.pickle_file_name, 'wb')
		)

		#Preprocess the data
		voltages, capas = preproc.ac_qv_preproc(
			voltages, capas,
			self.preproc_param['scale'], 
			self.preproc_param['vg_shift']
		)
		
		# Only expand the dim if the number of dimension is 1
		origin_input = np.expand_dims(
			voltages, axis=1) if  voltages.ndim == 1 else voltages
		adjoint_label = np.expand_dims(
			capas, axis=1) if capas.ndim == 1 else capas		

		# Create adjoint_input data
		adjoint_input = np.ones((origin_input.shape[0], 1))

		# Set the data type to np float for origin input, adjoint input, adjoint label
		origin_input = origin_input.astype(np.float32)
		adjoint_input = adjoint_input.astype(np.float32)
		adjoint_label = adjoint_label.astype(np.float32)
		
		# Write to database
		data_reader.write_db(
			'minidb', db_name, 
			[origin_input, adjoint_input, adjoint_label]
		)
		self.input_data_store[data_tag] = [db_name, origin_input.shape[0]]
		preproc.restore_voltages(
			self.preproc_param['scale'],
			self.preproc_param['vg_shift'],
			voltages
		)


	def build_nets(
		self,
		hidden_dims, 
		batch_size=1,
		optim_method = 'AdaGrad',
		optim_param = {'alpha':0.01, 'epsilon':1e-4},
	):
		assert len(self.input_data_store) > 0, 'Input data store is empty.'
		assert 'train' in self.input_data_store, 'Missing training data.'
		self.batch_size = batch_size
		# Build the date reader net for train net
		input_data_train = data_reader.build_input_reader(
			self.model, 
			self.input_data_store['train'][0], 
			'minidb', 
			['origin_input', 'adjoint_input', 'label'], 
			batch_size=batch_size,
			data_type='train',
		)

		if 'eval' in self.input_data_store:
			# Build the data reader net for eval net
			input_data_eval = data_reader.build_input_reader(
				self.model, 
				self.input_data_store['eval'][0], 
				'minidb', 
				['origin_input', 'adjoint_input'], 
				batch_size=batch_size,
				data_type='eval',
			)

		# Build the computational nets
		# Create train net
		self.model.input_feature_schema.origin_input.set_value(
			input_data_train[0].get(), unsafe=True)
		self.model.input_feature_schema.adjoint_input.set_value(
			input_data_train[1].get(), unsafe=True)
		self.model.trainer_extra_schema.label.set_value(
			input_data_train[2].get(), unsafe=True)

		self.origin_pred, self.adjoint_pred, self.loss = build_adjoint_mlp(
			self.model,
			input_dim = self.input_dim,
			hidden_dims = hidden_dims,
			output_dim = self.output_dim,
			optim=_build_optimizer(
				optim_method, optim_param),
		)

		train_init_net, train_net = instantiator.generate_training_nets(self.model)
		workspace.RunNetOnce(train_init_net)
		workspace.CreateNet(train_net)
		self.net_store['train_net'] = train_net

		pred_net = instantiator.generate_predict_net(self.model)
		workspace.CreateNet(pred_net)
		self.net_store['pred_net'] = pred_net
		
		if 'eval' in self.input_data_store:
			# Create eval net
			self.model.input_feature_schema.origin_input.set_value(
				input_data_eval[0].get(), unsafe=True)
			self.model.input_feature_schema.adjoint_input.set_value(
				input_data_eval[1].get(), unsafe=True)
			self.model.trainer_extra_schema.label.set_value(
				input_data_eval[2].get(), unsafe=True)
			eval_net = instantiator.generate_eval_net(self.model)
			workspace.CreateNet(eval_net)
			self.net_store['eval_net'] = eval_net


	def train_with_eval(
		self, 
		num_epoch=1,
		report_interval=0,
		eval_during_training=False,

	):
		''' Fastest mode: report_interval = 0
			Medium mode: report_interval > 0, eval_during_training=False
			Slowest mode: report_interval > 0, eval_during_training=True
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
			num_iter = num_epoch*num_batch_per_epoch
			workspace.RunNet(
				train_net, 
				num_iter=num_iter
			)
			
		print('>>> Saving test model')

		exporter.save_net(
			self.net_store['pred_net'], 
			self.model, 
			self.model_name+'_init', self.model_name+'_predict'
		)


	def draw_nets(self):
		for net_name in self.net_store:
			net = self.net_store[net_name]
			graph = net_drawer.GetPydotGraph(net.Proto().op, rankdir='TB')
			with open(net.Name() + ".png",'wb') as f:
				f.write(graph.create_png())
				

	def predict_qs(self, voltages):
		# requires voltages is an numpy array of size 
		# (batch size, input_dimension)
		# the first dimension is Vg and the second dimenstion is Vd

		# preprocess the origin input and create adjoint input
		# voltages array is unchanged 		
		if len(self.preproc_param) == 0:
			self.preproc_param = pickle.load(
				open(self.pickle_file_name, "rb" )
			)
		dummy_qs = np.zeros(voltages[0].shape[0])
		voltages, dummy_qs = preproc.ac_qv_preproc(
			voltages, dummy_qs, 
			self.preproc_param['scale'], 
			self.preproc_param['vg_shift']
		)
		adjoint_input = np.ones((voltages[0].shape[0], 1))
		
		# Expand dimensions of input and set data type of inputs
		origin_input = np.expand_dims(
			voltages, axis=1)
		origin_input = origin_input.astype(np.float32)
		adjoint_input = adjoint_input.astype(np.float32)

		workspace.FeedBlob('DBInput_train/origin_input', origin_input)
		workspace.FeedBlob('DBInput_train/adjoint_input', adjoint_input)
		pred_net = self.net_store['pred_net']
		workspace.RunNet(pred_net)

		qs = np.squeeze(schema.FetchRecord(self.origin_pred).get())
		gradients = np.squeeze(schema.FetchRecord(self.adjoint_pred).get())
		restore_integral_func, restore_gradient_func = preproc.get_restore_q_func( 
			self.preproc_param['scale'], 
			self.preproc_param['vg_shift']
		)
		original_qs = restore_integral_func(qs)
		original_gradients = restore_gradient_func(gradients)
		preproc.restore_voltages(
			self.preproc_param['scale'],
			self.preproc_param['vg_shift'],
			voltages
		)
		return qs, original_qs, gradients, original_gradients

	def plot_loss_trend(self):
		plt.plot(self.reports['epoch'], self.reports['train_loss'])
		if len(self.reports['eval_loss']) > 0:
			plt.plot(self.reports['epoch'], self.reports['eval_loss'], 'r--')
		plt.show()

	


	
# --------------------------------------------------------
# ----------------   Global functions  -------------------
# --------------------------------------------------------

def predict_qs(model_name, voltages):
	workspace.ResetWorkspace()

	# requires voltages is an numpy array of size 
	# (batch size, input_dimension)
	# the first dimension is Vg and the second dimenstion is Vd

	# preprocess the origin input and create adjoint input
	preproc_param = pickle.load(
			open(model_name+'_preproc_param.p', "rb" )
		)
	dummy_qs = np.zeros(voltages[0].shape[0])
	voltages, dummy_qs = preproc.ac_qv_preproc(
		voltages, dummy_qs, 
		preproc_param['scale'], 
		preproc_param['vg_shift']
	)
	adjoint_input = np.ones((voltages[0].shape[0], 1))
	
	# Expand dimensions of input and set data type of inputs
	origin_input = np.expand_dims(
		voltages, axis=1)
	origin_input = origin_input.astype(np.float32)
	adjoint_input = adjoint_input.astype(np.float32)

	workspace.FeedBlob('DBInput_train/origin_input', voltages)
	workspace.FeedBlob('DBInput_train/adjoint_input', adjoint_input)
	pred_net = exporter.load_net(model_name+'_init', model_name+'_predict')
	workspace.RunNet(pred_net)

	qs = np.squeeze(schema.FetchBlob('prediction'))
	gradients = np.squeeze(schema.FetchBlob('adjoint_prediction'))
	restore_integral_func, restore_gradient_func = preproc.get_restore_q_func( 
		preproc_param['scale'], 
		preproc_param['vg_shift']
	)
	original_qs = restore_integral_func(qs)
	original_gradients = restore_gradient_func(gradients)
	preproc.restore_voltages(
		self.preproc_param['scale'],
		self.preproc_param['vg_shift'],
		voltages
	)
	return qs, original_qs, gradients, original_gradients


def plot_iv( 
	vg, vd, ids, 
	vg_comp = None, vd_comp = None, ids_comp = None,
	styles = ['vg_major_linear', 'vd_major_linear', 'vg_major_log', 'vd_major_log']
):
	if 'vg_major_linear' in styles:
		visualizer.plot_linear_Id_vs_Vd_at_Vg(
			vg, vd, ids, 
			vg_comp = vg_comp, vd_comp = vd_comp, ids_comp = ids_comp,
		)
	if 'vd_major_linear' in styles:
		visualizer.plot_linear_Id_vs_Vg_at_Vd(
			vg, vd, ids, 
			vg_comp = vg_comp, vd_comp = vd_comp, ids_comp = ids_comp,
		)
	if 'vg_major_log' in styles:
		visualizer.plot_log_Id_vs_Vd_at_Vg(
			vg, vd, ids, 
			vg_comp = vg_comp, vd_comp = vd_comp, ids_comp = ids_comp,
		)
	if 'vd_major_log' in styles:
		visualizer.plot_log_Id_vs_Vg_at_Vd(
			vg, vd, ids, 
			vg_comp = vg_comp, vd_comp = vd_comp, ids_comp = ids_comp,
		)

def _build_optimizer(optim_method, optim_param):
	if optim_method == 'AdaGrad':
		optim = optimizer.AdagradOptimizer(**optim_param)
	elif optim_method == 'SgdOptimizer':
		optim = optimizer.SgdOptimizer(**optim_param)
	elif optim_method == 'Adam':
		optim = optimizer.AdamOptimizer(**optim_param)
	else:
		raise Exception(
			'Did you foget to implement {}?'.format(optim_method))
	return optim

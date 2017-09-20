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
import visualizer
# import logging
import matplotlib.pyplot as plt

class DCModel:
	def __init__(
		self, 
		model_name,
		sig_input_dim=1,
		tanh_input_dim=1,
		output_dim=1,
	):	
		self.model_name = model_name
		self.model = init_model_with_schemas(
			model_name, sig_input_dim, tanh_input_dim, output_dim)
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
		data_arrays are in the order of sig_input, tanh_input, and label
		'''
		assert len(data_arrays) == 3, 'Incorrect number of input data'
		# number of examples and same length assertion
		num_example = len(data_arrays[0])
		for data in data_arrays[1:]:
			assert len(data) == num_example, 'Mismatch dimensions'
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
		pickle.dump(
			self.preproc_param, 
			open(self.pickle_file_name, 'wb')
		)
		preproc_data_arrays = preproc.dc_iv_preproc(
			data_arrays[0], data_arrays[1], data_arrays[2], 
			self.preproc_param['scale'], 
			self.preproc_param['vg_shift'], 
			slope=self.preproc_param['preproc_slope'],
			threshold=self.preproc_param['preproc_threshold']
		)
		# Only expand the dim if the number of dimension is 1
		preproc_data_arrays = [np.expand_dims(
			x, axis=1) if x.ndim == 1 else x for x in preproc_data_arrays]
		# Write to database
		data_reader.write_db('minidb', db_name, *preproc_data_arrays)
		self.input_data_store[data_tag] = [db_name, num_example]

	def build_nets(
		self,
		hidden_sig_dims, 
		hidden_tanh_dims,
		batch_size=1,
		weight_optim_method = 'AdaGrad',
		weight_optim_param = {'alpha':0.005, 'epsilon':1e-4},
		bias_optim_method = 'AdaGrad',
		bias_optim_param = {'alpha':0.05, 'epsilon':1e-4},
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

		# Build the computational nets
		# Create train net
		self.model.input_feature_schema.sig_input.set_value(
			input_data_train[0].get(), unsafe=True)
		self.model.input_feature_schema.tanh_input.set_value(
			input_data_train[1].get(), unsafe=True)
		self.model.trainer_extra_schema.label.set_value(
			input_data_train[2].get(), unsafe=True)

		self.pred, self.loss = build_pinn(
			self.model,
			sig_net_dim=hidden_sig_dims,
			tanh_net_dim=hidden_tanh_dims,
			weight_optim=_build_optimizer(
				weight_optim_method, weight_optim_param),
			bias_optim=_build_optimizer(
				bias_optim_method, bias_optim_param),
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
				

	def predict_ids(self, vg, vd):
		# preproc the input
		vg = vg.astype(np.float32)
		vd = vd.astype(np.float32)
		if len(self.preproc_param) == 0:
			self.preproc_param = pickle.load(
				open(self.pickle_file_name, "rb" )
			)
		dummy_ids = np.zeros(len(vg))
		preproc_data_arrays = preproc.dc_iv_preproc(
			vg, vd, dummy_ids, 
			self.preproc_param['scale'], 
			self.preproc_param['vg_shift'], 
			slope=self.preproc_param['preproc_slope'],
			threshold=self.preproc_param['preproc_threshold']
		)
		_preproc_data_arrays = [np.expand_dims(
			x, axis=1) for x in preproc_data_arrays]
		workspace.FeedBlob('DBInput_train/sig_input', _preproc_data_arrays[0])
		workspace.FeedBlob('DBInput_train/tanh_input', _preproc_data_arrays[1])
		pred_net = self.net_store['pred_net']
		workspace.RunNet(pred_net)

		_ids = np.squeeze(schema.FetchRecord(self.pred).get())
		restore_id_func = preproc.get_restore_id_func( 
			self.preproc_param['scale'], 
			self.preproc_param['vg_shift'], 
			slope=self.preproc_param['preproc_slope'],
			threshold=self.preproc_param['preproc_threshold']
		)
		ids = restore_id_func(_ids, preproc_data_arrays[0])
		return _ids, ids

	def plot_loss_trend(self):
		plt.plot(self.reports['epoch'], self.reports['train_loss'])
		if len(self.reports['eval_loss']) > 0:
			plt.plot(self.reports['epoch'], self.reports['eval_loss'], 'r--')
		plt.show()

# --------------------------------------------------------
# ----------------   Global functions  -------------------
# --------------------------------------------------------

def plot_iv( 
	vg, vd, ids, 
	vg_comp = None, vd_comp = None, ids_comp = None,
	styles = ['vg_major_linear', 'vd_major_linear', 'vg_major_log', 'vd_major_log']
):
	fid = 0
	if 'vg_major_linear' in styles:
		plt.figure(fid)
		fid += 1
		visualizer.plot_linear_Id_vs_Vd_at_Vg(
			vg, vd, ids, 
			vg_comp = vg_comp, vd_comp = vd_comp, ids_comp = ids_comp,
		)
	if 'vd_major_linear' in styles:
		plt.figure(fid)
		fid += 1
		visualizer.plot_linear_Id_vs_Vg_at_Vd(
			vg, vd, ids, 
			vg_comp = vg_comp, vd_comp = vd_comp, ids_comp = ids_comp,
		)
	if 'vg_major_log' in styles:
		plt.figure(fid)
		fid += 1
		visualizer.plot_log_Id_vs_Vd_at_Vg(
			vg, vd, ids, 
			vg_comp = vg_comp, vd_comp = vd_comp, ids_comp = ids_comp,
		)
	if 'vd_major_log' in styles:
		plt.figure(fid)
		fid += 1
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


		 
			
			
			
			

import caffe2_paths
import os
from caffe2.python import (
	workspace, layer_model_helper, schema, optimizer, net_drawer
)
import caffe2.python.layer_model_instantiator as instantiator
import numpy as np
from pinn_lib import build_pinn, init_model_with_schemas
from data_reader import write_db, build_input_reader
from parser import read_dc_iv_mdm

class DCModel:
	def __init__(
		self, 
		model_name, 
		hidden_sig_dims, 
		hidden_tanh_dims,
		train_batch_size,
		test_batch_size,
		transfer_before_interconnect=False,
		interconnect_method,
		inner_embed_dim=[],
		train_file_name=None,
		eval_file_name=None,
	):
	
	@property
	def input_data(self):
		return input_data

	@input_data.setter
	def input_data(self, data_arrays=[], file_name_lst=[]):
		if len(data_arrays) > 0:
			print('Read into data directly from data arrays... ' + 
				'Note: data are in the order of vg, vd, and id')
		elif len(file_name_lst) > 0:
			print('Read in the data from data file...')
			_, file_ext = os.path.splitext(file_name)
			if file_ext == 'mdm':
				data_arrays = read_dc_iv_mdm(train_file_name)
			else file_ext == 'csv':
				data_arrays = read
		


	def build_(
		self,
		train_file_name,
		test_file_name,
		db_name, 
		batch_size = 100, 
		transfer_before_interconnect = False, 
		interconnect_method = 'Add', 
		inner_embed_dim = []
	):
		self.train_file_name = train_file_name
		vg, vd, label = read_dc_iv_mdm(train_file_name)
		self.sig_net_dim = [vg.shape[1]] + self.sig_net_dim
		self.tanh_net_dim = [vd.shape[1]] + self.tanh_net_dim
		self.pred_dim = id_label.shape[1]
		# self.test_data_file = test_file_name
		if not os.path.isfile (db_name):
			print ("Creating a new database.")
			# preproc
			write_db('minidb', db_name + '_train', vg, vd, label)
		else:
			# read out from pickle
			print ("Reading from existing database.")

		self.model = init_model_with_schemas(
			self.model_name, 
			self.sig_net_dim[0], 
			self.tanh_net_dim[0],
			self.pred_dim
		)
		# Need these for plotting
		input_1, input_2, label = add_two_inputs_and_label(
			self.model, 
			db_name, 'minidb', 'sig_input', 'tanh_input', 
			batch_size)
		self._build_model(
			label, 
			transfer_before_interconnect, 
			interconnect_method, 
			inner_embed_dim
		)

	def _build_model(
		self, 
		label, 
		transfer_before_interconnect, 
		interconnect_method, 
		inner_embed_dim
	):
		if (interconnect_method == 'Add'): 
			self.inner_embed_dim[0] = self.sig_net_input[0]
		else:
			assert len(inner_embed_dim) > 0, 'Inner_embed_dim cannot be empty'
			self.inner_embed_dim = inner_embed_dim

		self.pred, self.loss = build_pinn(
			self.model, 
			label, 
			sig_net_dim = self.sig_net_dim, 
			tanh_net_dim = self.tanh_net_dim, 
			inner_embed_dim = self.inner_embed_dim, 
			tranfer_before_interconnect = transfer_before_interconnect, 
			interconnect_method = interconnect_method
		)
	
	def _feed_test_data(self, test_file_name):
		X_sig, Y_tanh, label = read_dc_iv_mdm(test_data_file)
		# same as training
		# schema.FeedRecord(self.__model.input_feature_schema, 
		# 	[X_sig, Y_tanh])
		return label

	def test_model(self, test_file_name):
		label = self._feed_test_data(test_file_name)
		eval_net = instantiator.generate_eval_net(self.__model)
		workspace.CreateNet(eval_net)
		workspace.RunNet(eval_net.Proto().name)
		eval_loss = schema.FetchRecord(self.loss)
		eval_pred = schema.FetchRecord(self.pred)
		return eval_loss, eval_pred

	def train_and_test(self, iterations = 1000, loss_reports = 1):
		num_iter = iterations / loss_reports
		train_init_net, train_net = instantiator.generate_training_nets(self.__model)
		workspace.RunNetOnce(train_init_net)
		workspace.CreateNet(train_net)
		for i in range(loss_reports):
			print ("------------")
			workspace.RunNet(train_net, num_iter = int(num_iter))
			print(schema.FetchRecord(self.__loss).get())
		print ("------------")
		self.test_model()
	
		
		 
			
			
			
			

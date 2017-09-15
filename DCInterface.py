import caffe2_paths
import os
from caffe2.python import (
	workspace, layer_model_helper, schema, optimizer, net_drawer
)
import caffe2.python.layer_model_instantiator as instantiator
import numpy as np
from pinn_lib import build_pinn, init_model_with_schemas
from preproc import write_db, add_input_and_label
from parser import read_dc_iv_mdm

class DCModel:

	def __init__(self,model_name, hidden_dim_sig, hidden_dim_tan):
		self.__model_name = model_name
		self.__model = None
		self.__test_data_file = None
		self.__sig_net_dim = hidden_dim_sig
		self.__tanh_net_dim = hidden_dim_tan
		self.__inner_embed_dim = []
		self.__pred_dim = 0
		self.__pred = 0
		self.__loss = 0		


	def build_model_with_input (self, test_file_name, db_name, 
		train_file_name = None, batch_size = 100, 
		transfer_before_interconnect = False, 
		interconnect_method = 'Add', inner_embed_dim = [0]):
		self.__test_data_file = test_file_name
		if not os.path.isfile (db_name):
			assert train_file_name is not None,"Database does not exist and input file not specified."
			print ("Creating a new database.")
			vg, vd, label = read_dc_iv_mdm(train_file_name)
			self.__sig_input_dim = [vg.ndim] + self.__sig_input_dim
			self.__tanh_input_dim = [vd.ndim] + self.__tanh_input_dim
			self.__pred_dim = id_label.ndim
			write_db('minidb', db_name, vg, vd, label)
		else:
			print ("Reading from existing database.")
		self.__model = init_model_with_schemas(self.__model_name, 
			self.__sig_input_dim[0], self.__tanh_input_dim[0],
			self.__pred_dim)
		input_1, input_2, label = add_input_and_label (self.__model, 
			db_name, 'minidb', 'sig_input', 'tanh_input', 
			batch_size)
		self.__build_model(label, transfer_before_interconnect, 
				interconnect_method, inner_embed_dim)

	def __build_model (self, label, transfer_before_interconnect, 
			interconnect_method, inner_embed_dim):
		if (interconnect_method == 'Add'): 
			self.__inner_embed_dim[0] = sig_input[0]
		else:
			self.__inner_embed_dim = inner_embed_dim
		self.__pred, self.__loss = build_pinn (self.__model, label, 
			sig_net_dim = self.__sig_net_dim, tanh_net_dim = 
			self.__tanh_net_dim, inner_embed_dim = 
			self.__inner_embed_dim, tranfer_before_interconnect = 
			transfer_before_interconnect, interconnect_method = 
			interconnect_method)
	
	def __feed_test_data (self):
		X_sig, Y_tanh, label = read_dc_iv_mdm(self.__test_data_file)
		schema.FeedRecord(self.__model.input_feature_schema, 
			[X_sig, Y_tanh])
		return label

	def test_model (self):
		label = self.__feed_test_data()
		eval_net = instantiator.generate_eval_net(self.__model)
		workspace.CreateNet(eval_net)
		workspace.RunNet(eval_net.Proto().name)
		eval_loss = schema.FetchRecord(self.__loss)
		eval_pred = schema.FetchRecord(self.__pred)
		print (eval_loss)
		print (eval_pred)
		return eval_loss, eval_pred

	def train_and_test (self, iterations = 1000, loss_reports = 1):
		num_iter = iterations / loss_repors
		train_init_net, train_net = instantiator.generate_training_nets(			self.__model)
		workspace.RunNetOnce(train_init_net)
		workspace.CreateNet(train_net)
		for i in range (loss_reports):
			print ("------------")
			workspace.RunNet(train_net, num_iter = int( num_iter))
			print(schema.FetchRecord(self.__loss).get())
		print ("------------")
		self.test_model()
	
		
		 
			
			
			
			

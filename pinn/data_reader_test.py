import sys
sys.path.append('../')
import caffe2_paths
import numpy as np
import os
from caffe2.python import workspace, layer_model_helper, schema, core, utils, workspace
import data_reader
import unittest

class TestDataReader(unittest.TestCase):
	def test_write_db(self):
		num_example = 10
		batch_size = 3
		np.random.seed(42)
		features_expected = np.random.rand(num_example, 2).astype('float32')
		labels_expected = np.random.rand(num_example, 2).astype('float32')
		workspace.ResetWorkspace()
		if os.path.isfile('test.db'):
			os.remove("test.db")
		data_reader.write_db('minidb', 'test.db', 
			[features_expected, features_expected, labels_expected])
		init_net = core.Net("example_reader_init")
		net_proto = core.Net("example_reader")
		dbreader = init_net.CreateDB(
			[], "dbreader", db="test.db", db_type="minidb")
		net_proto.TensorProtosDBInput(
			[dbreader], ["feature1", "feature2", "labels"], batch_size=batch_size)
		workspace.RunNetOnce(init_net)
		workspace.CreateNet(net_proto)

		for i in range(num_example//batch_size):
			workspace.RunNet(net_proto.Proto().name)
			self.assertTrue(
				np.array_equal(workspace.FetchBlob("feature1"), 
					features_expected[i * batch_size : (i + 1) * batch_size])
			)
			self.assertTrue(
				np.array_equal(workspace.FetchBlob("labels"),
					labels_expected[i * batch_size : (i + 1) * batch_size]))

if __name__ == '__main__':
    unittest.main()

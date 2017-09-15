import caffe2_paths
import numpy as np
import os
from caffe2.python import workspace, layer_model_helper, schema, core, utils, workspace
import parser
import preproc
import unittest

class TestPreproc(unittest.TestCase):
	def test_write_db(self):
		num_example = 10
		batch_size = 3
		np.random.seed(42)
		features_expected = np.random.rand(num_example, 2).astype('float32')
		labels_expected = np.random.rand(num_example, 2).astype('float32')
		workspace.ResetWorkspace()
		if os.path.isfile('test.db'):
			os.remove("test.db")
		preproc.write_db('minidb', 'test.db', 
			features_expected, features_expected, labels_expected)
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
		
	def test_dc_iv_preproc(self):
		vg, vd, ids = parser.read_dc_iv_mdm("./HEMT_bo/Id_vs_Vd_at_Vg.mdm")
		scale, vg_shift = preproc.compute_dc_meta(vg, vd, ids)
		preproc_vg, preproc_vd, preproc_id, restore_func = preproc.dc_iv_preproc(
			vg, vd, ids, scale, vg_shift
		)
		self.assertTrue(
			len(preproc_vg) > 0 and 
			len(preproc_vd) > 0 and 
			len(preproc_vg) > 0
		)
		ori_vg, ori_vd, ori_id = restore_func(preproc_vg, preproc_vd, preproc_id)
		np.testing.assert_array_almost_equal(ids, ori_id, 10)
		np.testing.assert_array_almost_equal(vd,  ori_vd, 10)
		np.testing.assert_array_almost_equal(vg,  ori_vg, 10)

if __name__ == '__main__':
    unittest.main()
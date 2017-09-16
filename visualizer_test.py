import caffe2_paths
import numpy as np
import os
from caffe2.python import workspace, layer_model_helper, schema, core, utils, workspace
import parser
import preproc
import unittest
import visualizer

class TestVisualizer(unittest.TestCase):
	def test_visualizer(self):
		vg, vd, ids = parser.read_dc_iv_mdm("./HEMT_bo/Id_vs_Vd_at_Vg.mdm")
		scale, vg_shift = preproc.compute_dc_meta(vg, vd, ids)
		preproc_vg, preproc_vd, preproc_id, restore_func = preproc.dc_iv_preproc(
			vg, vd, ids, scale, vg_shift
		)
		data_out = np.array([preproc_vg, preproc_vd, preproc_id])
		write_csv_file('after_pre-process.csv', data_out, 'Row 1: Vg, Row 2: Vd, Row 3: Id')



if __name__ == '__main__':
    unittest.main()

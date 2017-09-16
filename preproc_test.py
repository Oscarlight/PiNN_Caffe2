import caffe2_paths
import numpy as np
import parser
import preproc
import unittest

class TestPreproc(unittest.TestCase):		
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

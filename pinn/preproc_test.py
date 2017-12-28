import numpy as np
import parser
import preproc
import unittest

class TestPreproc(unittest.TestCase):		
	def test_dc_iv_preproc(self):
		vg = np.array([[0.1, 0.8], [0.2, 0.4], [0.4, 0.2]])
		vd = np.array([0, 0.8, 1.6])
		ids = np.array([1e-6, 1e-5, 1e-4])
		scale, vg_shift = preproc.compute_dc_meta(vg, vd, ids)
		print(scale, vg_shift)
		preproc_vg, preproc_vd, preproc_id = preproc.dc_iv_preproc(
			vg, vd, ids, scale, vg_shift
		)
		self.assertTrue(
			len(preproc_vg) > 0 and 
			len(preproc_vd) > 0 and 
			len(preproc_vg) > 0
		)
		print(preproc_vg)
		restore_id_func,_ = preproc.get_restore_id_func(scale)
		ori_id = restore_id_func(preproc_id)
		np.testing.assert_array_almost_equal(ids, ori_id, 10)

if __name__ == '__main__':
    unittest.main()

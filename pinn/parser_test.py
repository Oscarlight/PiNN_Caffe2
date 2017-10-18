import unittest
import parser
import visualizer

class TestPreproc(unittest.TestCase):
	def test_read_dc_iv_mdm(self):
		parser.read_dc_iv_mdm("../HEMT_bo/Id_vs_Vd_at_Vg.mdm")

	def test_read_ac_s_par_mdm(self):
		s11arr,s12arr,s21arr,s22arr,freq,vg,vd,ids = parser.read_s_par_mdm('../HEMT_bo/s_at_f_vs_VgVd.mdm')
		visualizer.plot_linear_Id_vs_Vd_at_Vg(vg, vd, s21arr, yLabel = 'S$_{21}$')

if __name__ == '__main__':
    unittest.main()
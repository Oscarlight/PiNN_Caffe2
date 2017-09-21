import unittest
import parser

class TestPreproc(unittest.TestCase):
	def test_read_dc_iv_mdm(self):
		parser.read_dc_iv_mdm("./HEMT_bo/Id_vs_Vd_at_Vg.mdm")

if __name__ == '__main__':
    unittest.main()
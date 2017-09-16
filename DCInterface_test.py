from DCInterface import DCModel as DCModel

def DCInterface_test (model_name, test_file_name, db_name, train_file_name):
	model_test = DCModel(model_name, [1], [1])
	model_test.build_model_with_input(test_file_name, db_name, train_file_name)
	model.train_and_test()


def main ():
	DCInterface_test(
		'model_test',
		'./HEMT_bo/Id_vs_Vd_at_Vg.mdm',
		'db_test.db',
		'./HEMT_bo/Id_vs_Vd_at_Vg.mdm'
	)

if __name__ == '__main__':
	main()

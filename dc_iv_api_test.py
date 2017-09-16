from dc_iv_api import DCModel

dc_model = DCModel(
	'hemt_dc_test_1', 
	train_file_name='./HEMT_bo/Id_vs_Vd_at_Vg.mdm',
	eval_file_name='./HEMT_bo/Id_vs_Vg_at_Vd.mdm',
)

dc_model.build_nets(
	hidden_sig_dims=[3, 3],
	hidden_tanh_dims=[3, 3],
	train_batch_size = 100,
	eval_batch_size = 100,
)



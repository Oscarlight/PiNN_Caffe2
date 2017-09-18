from dc_iv_api import DCModel

# TODO: Test on
# './HEMT_bo/Id_vs_Vd_at_Vg.mdm'
# './HEMT_bo/Id_vs_Vg_at_Vd.mdm'

dc_model = DCModel(
	'hemt_dc_test_2', 
	train_file_name='./HEMT_bo/DC_IV.csv',
)

dc_model.build_nets(
	hidden_sig_dims=[3, 1],
	hidden_tanh_dims=[3, 1],
	batch_size=10,
	optim_param = {'alpha':0.01, 'epsilon':1e-4} 
)
# If you want to take a look at the nets
dc_model.draw_nets()

dc_model.train_with_eval(
	num_epoch=1000, 
	report_interval=100,
)

dc_model.plot_loss_trend()


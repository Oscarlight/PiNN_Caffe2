## current best one (note: the definition of hidden layer changed):
# python transixor_trainer.py transiXOR_Models/bise_ext_sym_h264_0 \
# -hidden 64 -layer 2 -mls 500 -batchsize 2048 -lossfunct scaled_l1 -epoch 500000

## add neg grad penalty
python transixor_trainer.py transiXOR_Models/bise_ext_sym_h264_neggrad_0 \
	-hidden 64 64 -mls 500 -batchsize 2048 -lossfunct scaled_l1 \
	-epoch 5000 -neg_grad_mag 1 -report 1000 -lr 0.01

python transixor_trainer.py transiXOR_Models/bise_ext_sym_h264_neggrad_1 \
	-hidden 64 64 -mls 500 -batchsize 2048 -lossfunct scaled_l1 \
	-epoch 5000 -neg_grad_mag 10 -report 1000  -lr 0.01

python transixor_trainer.py transiXOR_Models/bise_ext_sym_h264_neggrad_2 \
	-hidden 64 64 -mls 500 -batchsize 2048 -lossfunct scaled_l1 \
	-epoch 5000 -neg_grad_mag 100 -report 1000 -lr 0.01

python transixor_trainer.py transiXOR_Models/bise_ext_sym_h264_neggrad_3 \
	-hidden 64 64 -mls 500 -batchsize 2048 -lossfunct scaled_l1 \
	-epoch 5000 -neg_grad_mag 1000 -report 1000 -lr 0.01

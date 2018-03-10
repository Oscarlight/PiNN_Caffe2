# python transixor_trainer.py transiXOR_Models/bise_h18_0 -hidden 8 -layer 1
# python transixor_trainer.py transiXOR_Models/bise_h28_0 -hidden 8 -layer 2
# python transixor_trainer.py transiXOR_Models/bise_h116_0 -hidden 16 -layer 1
# python transixor_trainer.py transiXOR_Models/bise_h216_0 -hidden 16 -layer 2

# python transixor_trainer.py transiXOR_Models/bise_h18_1 -hidden 8 -layer 1 -mls 500
# python transixor_trainer.py transiXOR_Models/bise_h28_1 -hidden 8 -layer 2 -mls 500
# python transixor_trainer.py transiXOR_Models/bise_h116_1 -hidden 16 -layer 1 -mls 500
# python transixor_trainer.py transiXOR_Models/bise_h216_1 -hidden 16 -layer 2 -mls 500

## extend V to -0.1 to 0.3 V
# python transixor_trainer.py transiXOR_Models/bise_ext_h216_0 -hidden 16 -layer 2 -mls 500 -batchsize 2048 -lossfunct scaled_l2
# python transixor_trainer.py transiXOR_Models/bise_ext_h216_1 -hidden 16 -layer 2 -mls 50 -batchsize 2048 -lossfunct scaled_l1
# python transixor_trainer.py transiXOR_Models/bise_ext_h232_0 -hidden 32 -layer 2 -mls 5000 -batchsize 2048 -lossfunct scaled_l2
# python transixor_trainer.py transiXOR_Models/bise_ext_h232_1 -hidden 32 -layer 2 -mls 500 -batchsize 2048 -lossfunct scaled_l1
python transixor_trainer.py transiXOR_Models/bise_ext_sym_h264_0 -hidden 64 -layer 2 -mls 500 -batchsize 2048 -lossfunct scaled_l1 -epoch 5e5
python transixor_trainer.py transiXOR_Models/bise_ext_sym_h264_1 -hidden 64 -layer 2 -mls 1000 -batchsize 2048 -lossfunct scaled_l1 -epoch 5e5
python transixor_trainer.py transiXOR_Models/bise_ext_sym_h232_0 -hidden 32 -layer 2 -mls 1000 -batchsize 2048 -lossfunct scaled_l1 -epoch 5e5

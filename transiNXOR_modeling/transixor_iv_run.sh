python transixor_trainer.py transiXOR_Models/bise_h18 -hidden 8 -layer 1
python transixor_trainer.py transiXOR_Models/bise_h28 -hidden 8 -layer 2
python transixor_trainer.py transiXOR_Models/bise_h116 -hidden 16 -layer 1
python transixor_trainer.py transiXOR_Models/bise_h216 -hidden 16 -layer 2

python transixor_trainer.py transiXOR_Models/bise_h18 -hidden 8 -layer 1 -mls 500
python transixor_trainer.py transiXOR_Models/bise_h28 -hidden 8 -layer 2 -mls 500
python transixor_trainer.py transiXOR_Models/bise_h116 -hidden 16 -layer 1 -mls 500
python transixor_trainer.py transiXOR_Models/bise_h216 -hidden 16 -layer 2 -mls 500
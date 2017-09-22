# PiNN
A easy-to-use interface for efficient device compact modeling using neural neworks.

## Install
Install Caffe2 from https://github.com/Oscarlight/caffe2.git. The forked repo contains some new layers created for this project. The instruction about how to install caffe2 is in https://caffe2.ai.

In order to make advantage of multi-core CPU, you need to install NNPack (Following steps are for Ubuntu system):
(If you are using a virtual env, run the python related install in your virtual env)
1. Download Ninja 1.7.2 from https://github.com/ninja-build/ninja/releases
2. Enter the folder, run ./configure.py --bootstrap
3. Run ./configure.py && ./ninja ninja_test && ./ninja_test --gtest_filter=-SubprocessTest.SetWithLots to test
4. sudo install ninja /usr/bin/
5. [sudo] pip install --upgrade git+https://github.com/Maratyszcza/PeachPy
6. [sudo] pip install --upgrade git+https://github.com/Maratyszcza/confu
7. git clone https://github.com/Maratyszcza/NNPACK.git
8. cd NNPACK
9. confu setup
10. python ./configure.py
11. ninja

If you consider data reading is the bottleneck, you could install rocksdb from https://github.com/facebook/rocksdb/blob/master/INSTALL.md.

Remember to re-build caffe2 after install NNPack and RocksDB.

## What can it do?
- DC IV modeling using Pi-NN (Completed)
- AC IV and QV modeling using adjoint neural network (Coming soon)

## How to use it?
- DC IV APIs
  - DCModel class
  examples: hemt_example_1.py (more examples are coming soon)

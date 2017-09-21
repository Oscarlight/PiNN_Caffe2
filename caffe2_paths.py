import sys
import os
user_dir = os.environ['HOME'] 
# Malavika
if user_dir == '/home/malavika':
	sys.path.append(user_dir + '/caffe2/build/')
# Mingda
if user_dir == '/home/oscar':
	sys.path.append(user_dir + '/Documents/caffe2/build/')
if user_dir == '/Users/Mingda':
	sys.path.append(user_dir + '/Documents/Caffe2/caffe2/build/')
# Xiang

#Jashan
if user_dir == '/Users/jashansinghal':
	sys.path.append(user_dir + '/Box Sync/PINNGit/caffe2/build')
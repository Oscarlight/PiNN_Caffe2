import caffe2_paths
import os
from caffe2.python import (
	core, workspace, layer_model_helper, schema, optimizer, net_drawer
)
import caffe2.python.layer_model_instantiator as instantiator
from pinn.data_reader import write_db, build_input_reader
from pinn.adjoint_pinn_lib import (
	build_adjoint_pinn, init_model_with_schemas
)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import math
model = init_model_with_schemas('adjoint_pinn', 1, 1, 1)
# example data
def f(x,y):
    return math.tanh(5*y)*(1/(1+math.exp(0-5*x)))
x = y = np.arange(-1, 1, 0.2)
X, Y = np.meshgrid(x, y)
zs = np.array([f(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)
# np.gradient reverse the order
gy, gx = np.gradient(Z, 0.2, 0.2)

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#     linewidth=0, antialiased=False)
# plt.show()
# quit()
sig_input = np.expand_dims(np.ravel(X), axis=1)
tanh_input = np.expand_dims(np.ravel(Y), axis=1)
sig_adjoint_label = np.expand_dims(np.ravel(gx), axis=1)
tanh_adjoint_label = np.expand_dims(np.ravel(gy), axis=1)

sig_input = np.array(sig_input, dtype=np.float32)
tanh_input = np.array(tanh_input, dtype=np.float32)
sig_adjoint_label = np.array(sig_adjoint_label, dtype=np.float32)
tanh_adjoint_label = np.array(tanh_adjoint_label, dtype=np.float32)
adjoint_input = np.ones((len(sig_input),1), dtype=np.float32)
# create db if needed
db_name = 'adjoint_pinn.db'
if os.path.isfile(db_name):
	os.remove(db_name)
write_db('minidb', db_name, 
	[sig_input, tanh_input, adjoint_input, sig_adjoint_label, tanh_adjoint_label])
(sig_input, tanh_input, adjoint_input, 
	sig_adjoint_label, tanh_adjoint_label) = build_input_reader(
	model, db_name, 'minidb', 
	['sig_input', 'tanh_input', 'adjoint_input', 'sig_adjoint_label', 'tanh_adjoint_label'], 
	batch_size=100
)
model.input_feature_schema.sig_input.set_value(
	sig_input.get(), unsafe=True)
model.input_feature_schema.tanh_input.set_value(
	tanh_input.get(), unsafe=True)
model.input_feature_schema.adjoint_input.set_value(
	adjoint_input.get(), unsafe=True)
model.trainer_extra_schema.sig_loss_record.label.set_value(
	sig_adjoint_label.get(), unsafe=True)
model.trainer_extra_schema.tanh_loss_record.label.set_value(
	tanh_adjoint_label.get(), unsafe=True)
# Build model
(origin_pred, sig_adjoint_pred, 
	tanh_adjoint_pred, loss) = build_adjoint_pinn(
	model, 
	sig_net_dim=[10, 1], tanh_net_dim=[10, 1],
	weight_optim=optimizer.AdagradOptimizer(alpha=0.01, epsilon=1e-4,),
	bias_optim=optimizer.AdagradOptimizer(alpha=0.01, epsilon=1e-4,)
)

# Train the model
train_init_net, train_net = instantiator.generate_training_nets(model)
workspace.RunNetOnce(train_init_net)
workspace.CreateNet(train_net)
num_iter = 1000
eval_num_iter = 100
# loss_lst = []
for i in range(eval_num_iter):
	workspace.RunNet(train_net.Proto().name, num_iter=num_iter)
	print(schema.FetchRecord(loss).get())

X_pred = np.squeeze(schema.FetchRecord(model.input_feature_schema.sig_input).get()).reshape(X.shape)
Y_pred = np.squeeze(schema.FetchRecord(model.input_feature_schema.tanh_input).get()).reshape(X.shape)
Z_pred = np.squeeze(schema.FetchRecord(origin_pred).get()).reshape(X.shape)
gy_restore, gx_restore = np.gradient(Z_pred, 0.2, 0.2)
gx_pred = schema.FetchRecord(sig_adjoint_pred).get().reshape(X.shape)
gy_pred = schema.FetchRecord(tanh_adjoint_pred).get().reshape(X.shape)
gx_label =schema.FetchRecord(model.trainer_extra_schema.sig_loss_record.label).get().reshape(X.shape)
gy_label =schema.FetchRecord(model.trainer_extra_schema.tanh_loss_record.label).get().reshape(X.shape)
# visulization
fig1 = plt.figure()
ax1 = fig1.gca()
fig2 = plt.figure()
ax2 = fig2.gca()
fig3 = plt.figure()
ax3 = fig3.gca(projection='3d')
fig4 = plt.figure()
ax4 = fig4.gca(projection='3d')
ax1.contourf(X_pred, Y_pred, gx_label, cmap=cm.coolwarm,
    linewidth=0, antialiased=False)
ax2.contourf(X_pred, Y_pred, gx_pred, cmap=cm.coolwarm,
    linewidth=0, antialiased=False)
ax3.plot_surface(X_pred, Y_pred, Z, cmap=cm.coolwarm,
    linewidth=0, antialiased=False)
ax4.plot_surface(X_pred, Y_pred, Z_pred, cmap=cm.coolwarm,
    linewidth=0, antialiased=False)
plt.show()

# Eval
eval_net = instantiator.generate_eval_net(model)
graph = net_drawer.GetPydotGraph(eval_net.Proto().op, rankdir='TB')
with open(eval_net.Name() + ".png",'wb') as f:
	f.write(graph.create_png())
f = open('eval_net.txt','w')
f.write(str(eval_net.Proto()))
f.close()
# Predict
# pred_net = instantiator.generate_predict_net(model)
# graph = net_drawer.GetPydotGraph(pred_net.Proto().op, rankdir='TB')
# with open(pred_net.Name() + ".png",'wb') as f:
# 	f.write(graph.create_png())

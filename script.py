from run_example import run_example
import matplotlib.pyplot as plt
import os

path = './maxloss1e6/saved_data_5/'

try: 
	os.makedirs(path)
except OSError:
	if not os.path.isdir(path):
		raise

k1 = [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.15,0.2,0.25]
epochlist = []
train_losslist = []
eval_losslist = []
train_l1_metriclist = []
eval_l1_metriclist = []
train_scaled_l1_metriclist = []
eval_scaled_l1_metriclist = []

for k in k1:
	epoch,train_loss,eval_loss,train_l1_metric,eval_l1_metric,train_scaled_l1_metric,eval_scaled_l1_metric = run_example(k)
	epochlist.append(epoch)
	train_losslist.append(train_loss)
	eval_losslist.append(eval_loss)
	train_l1_metriclist.append(train_l1_metric)
	eval_l1_metriclist.append(eval_l1_metric)
	train_scaled_l1_metriclist.append(train_scaled_l1_metric)
	eval_scaled_l1_metriclist.append(eval_scaled_l1_metric)

f = open(path+"diff-alpha.csv", "w")
f.write("{},{},{},{},{},{},{},{}\n".format("alpha","epoch", "train_loss","eval_loss","train_l1_metric","eval_l1_metric",
						                   "train_scaled_l1_metric","eval_scaled_l1_metric" ))
for x in zip(k1,epochlist,train_losslist,eval_losslist,train_l1_metriclist,eval_l1_metriclist,
			 train_scaled_l1_metriclist,eval_scaled_l1_metriclist):
	f.write("{},{},{},{},{},{},{},{}\n".format(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]))
f.close()

plt.plot(
	k1, 
	train_losslist, 'r', 
	label='train error'
		)
plt.plot(
	k1, 
	train_scaled_l1_metriclist, 'b', 
	label='train_scaled_l1_metric'
)
plt.plot(
	k1, 
	train_l1_metriclist, 'g', 
	label='train_l1_metric'
)

plt.plot(
	k1, 
	eval_losslist, 'r--',
	label='eval error'
)
plt.plot(
	k1, 
	eval_scaled_l1_metriclist, 'b--',
	label='eval_scaled_l1_metric'
)
plt.plot(
	k1, 
	eval_l1_metriclist, 'g--', 
	label='eval_l1_metric'
)
plt.legend()
plt.savefig(path+"diff_alpha.png")
plt.close()





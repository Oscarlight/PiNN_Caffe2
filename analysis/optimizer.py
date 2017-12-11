import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
from pylab import rcParams
from matplotlib.ticker import AutoMinorLocator

FIGURE_SIZE = (6, 8)
FONT_SIZE = 24
LINE_WIDTH = 1
MAJOR_LABEL_SIZE = 19
MINOR_LABEL_SIZE = 0
rcParams['figure.figsize'] = FIGURE_SIZE
rcParams['axes.linewidth'] = LINE_WIDTH

xminorLocator = AutoMinorLocator()
yminorLocator = AutoMinorLocator()

# index -> max_loss_scale value
mapping = {1: 1e5, 2: 2e5, 3: 5e5, 4: 5e5, 5: 5e4,
       6: 1e4, 7: 5e3, 8: 1e3, 9: 5e2, 10:1e2}
data = {}
for file in glob.glob('../model_output_*/*_loss_trend.csv'):
	idx = file.split('_')[2].split('/')[0] + '_' + file.split('_')[4]
	data[idx] = pd.read_csv(file)

idx = '0_5'
loss1 = [data[idx]['epoch'],
		data[idx]['train_scaled_l1_metric'],
		data[idx]['eval_scaled_l1_metric'],
		data[idx]['train_l1_metric'],
		data[idx]['eval_l1_metric'],
		data[idx]['train_loss'],
		data[idx]['eval_loss']]
idx = '0_6'
loss2 = [data[idx]['epoch'],
		data[idx]['train_scaled_l1_metric'],
		data[idx]['eval_scaled_l1_metric'],
		data[idx]['train_l1_metric'],
		data[idx]['eval_l1_metric'],
		data[idx]['train_loss'],
		data[idx]['eval_loss']]

TYPE_ID = 1
fig, ax = plt.subplots()
plt.plot(loss1[0], loss1[TYPE_ID], color='#3498db', linewidth=2)
plt.plot(loss1[0], loss1[TYPE_ID+1], color="#e74c3c", linewidth=2)
plt.plot(loss2[0], loss2[TYPE_ID], color='#3498db', linewidth=2, linestyle='--')
plt.plot(loss2[0], loss2[TYPE_ID+1], color="#e74c3c", linewidth=2, linestyle='--')
plt.tick_params(axis='both', which='major', length=10, labelsize=MAJOR_LABEL_SIZE)
plt.tick_params(axis='both', which='minor', length=5, labelsize=MINOR_LABEL_SIZE)
ax.set_yscale('log')
ax.set_xscale('log')
plt.xlim([500, 2e7])

# ax.xaxis.set_minor_locator(xminorLocator)
# ax.yaxis.set_minor_locator(yminorLocator)
#
# plt.ylim([1e-2, 5])
# plt.savefig('loss_trend_scaled_l1.eps')
#
# plt.ylim([1e-3, 1])
# plt.savefig('loss_trend_l1.eps')
#
plt.ylim([1e-3, 1])
# plt.savefig('loss_trend.eps')
plt.show()
quit()

LAST_POINT = 999 # 1000000 epoch
for i, _ in mapping.items():
	if i != 4:
		temp_loss = []
		for j in range(5):
			idx = str(j) + '_' + str(i)
			temp_loss.append(
				[data[idx]['train_scaled_l1_metric'][LAST_POINT],
				 data[idx]['eval_scaled_l1_metric'][LAST_POINT],
				 data[idx]['train_l1_metric'][LAST_POINT],
				 data[idx]['eval_l1_metric'][LAST_POINT],
				 data[idx]['train_loss'][LAST_POINT],
				 data[idx]['eval_loss'][LAST_POINT]])
		loss.append(temp_loss)

sort_idx = np.argsort(max_loss_scale)
max_loss_scale = np.array(max_loss_scale)[sort_idx]
loss = np.array(loss)[sort_idx]

fig, ax = plt.subplots()
TYPE_ID=2
SCALE = 1000
for i in range(5):
	plt.scatter(max_loss_scale, loss[:,i,TYPE_ID]*SCALE, s = 50, color='#add8e6', edgecolor='none', alpha=0.7)
	plt.scatter(max_loss_scale, loss[:,i,TYPE_ID+1]*SCALE, s = 50, color='#e6bbad', edgecolor='none', alpha=0.7)
plt.plot(max_loss_scale, np.mean(loss[:,:,TYPE_ID], axis=1)*SCALE, color='#3498db', linewidth=2)
plt.plot(max_loss_scale, np.mean(loss[:,:,TYPE_ID+1], axis=1)*SCALE, color="#e74c3c", linewidth=2)
plt.tick_params(axis='both', which='major', length=10, labelsize=MAJOR_LABEL_SIZE)
plt.tick_params(axis='both', which='minor', length=5, labelsize=MINOR_LABEL_SIZE)
# ax.set_yscale('log')
ax.set_xscale('log')
plt.xlim([50, 1e6])
plt.ylim([0.5, 2.6])
# ax.xaxis.set_minor_locator(xminorLocator)
ax.yaxis.set_minor_locator(yminorLocator)
# plt.savefig('scaled_l1_metric.eps')
# plt.show()
plt.savefig('l1_metric.eps')


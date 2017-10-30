import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np

# index -> max_loss_scale value
mapping = {1: 1e5, 2: 2e5, 3: 5e5, 4: 5e5, 5: 5e4,
       6: 1e4, 7: 5e3, 8: 1e3, 9: 5e2, 10:1e2}
data = {}
for file in glob.glob('../model_output_4/*_loss_trend.csv'):
	data[int(file.split('_')[4])] = pd.read_csv(file)

max_loss_scale = []; loss = []
for i, mls in mapping.items():
	if i != 4:
		# plt.loglog(data[i]['epoch'], data[i]['train_scaled_l1_metric'],label=str(mls))
		# plt.loglog(data[i]['epoch'], data[i]['eval_scaled_l1_metric'],label=str(mls))
		max_loss_scale.append(mls)
		loss.append(
			[data[i]['train_scaled_l1_metric'],
			 data[i]['eval_scaled_l1_metric'],
			 data[i]['train_l1_metric'],
			 data[i]['eval_l1_metric']])

sort_idx = np.argsort(max_loss_scale)
max_loss_scale = np.array(max_loss_scale)[sort_idx]
loss = np.array(loss)[sort_idx]
for i in [0, 2]:
	plt.loglog(max_loss_scale, loss[:,i, 999])

# plt.legend()
plt.show()
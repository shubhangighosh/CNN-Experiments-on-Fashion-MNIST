import numpy, pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


f = open('tmp/conv1_filters.pkl', 'rb')
filters = pickle.load(f)

gs = gridspec.GridSpec(8, 8)

for k in range(32):
	filter_weights = []
	for i in range(3):
		filter_row = []
		for j in range(3):
			filter_row.append(filters[i][j][0][k])
		filter_weights.append(filter_row)
	

	#Plot filter_weights
	ax = plt.subplot(gs[k/8, k%8])
	imgplot = ax.imshow(filter_weights, cmap="binary")
	ax.set_xticks([])
	ax.set_yticks([])

plt.show()






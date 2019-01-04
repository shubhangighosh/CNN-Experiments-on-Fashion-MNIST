import numpy, pickle
import matplotlib.pyplot as plt


f = open('tmp/losses', 'r')

epochs = []
losses_train = []
losses_val = []
for line in f.readlines():
	epoch, train, val = line.strip().split()
	epochs.append(int(epoch))
	losses_train.append(float(train)/55000.0)
	losses_val.append(float(val)/5000.0)


plt.plot(epochs,losses_train, linewidth=2, markersize=12, label="Train Loss")
plt.plot(epochs,losses_val, linewidth=2, markersize=12, label="Validation Loss")
plt.legend()
plt.title("Loss vs Epochs")
plt.show()






import numpy, pickle
import matplotlib.pyplot as plt


f = open('tmp/fooling', 'r')

pixels = []
accs = []
for line in f.readlines():
	pixel, acc = line.strip().split()
	pixels.append(float(pixel))
	accs.append(float(acc))



plt.plot(pixels, accs, linewidth=2, markersize=12)
plt.title("Pixels Changed vs Test Accuracy")
plt.show()






import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
import pandas as pd, numpy as np
import matplotlib.gridspec as gridspec


labeled_images = pd.read_csv('train.csv')
images = labeled_images.iloc[:,1:-1]
X_train = X_train.reshape(-1, 28, 28, 1)


#Single Training Image
img=images.iloc[0].as_matrix().reshape((28,28))
plt.imshow(img,cmap='binary')
plt.savefig('real.png')
plt.show()





########################################################################################################
#Agumented Images

X_train_aug1 = iaa.Fliplr(1).augment_images(X_train)
X_train_aug2 = iaa.AverageBlur(k=(3, 4)).augment_images(X_train)
X_train_aug3 = iaa.Emboss(alpha=(0.8, 1.0), strength=(0, 3.0)).augment_images(X_train)
X_train_aug4 = iaa.Sharpen(alpha=(0.8, 0.9), lightness=(0.75, 1.5)).augment_images(X_train)
X_train_aug5 = iaa.AdditiveGaussianNoise(loc=0, scale=(5.0, 0.05*255)).augment_images(X_train)
X_train_aug6 = iaa.ContrastNormalization((2.0, 2.5)).augment_images(X_train)


for i in range(10):
	#visualize X_train[i], X_train_aug1[i], etc
	gs = gridspec.GridSpec(3, 3)

	ax = plt.subplot(gs[0,0])
	imgplot = ax.imshow(X_train[i].reshape((28,28)), cmap="binary")
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_title('Original') 
	ax.title.set_size(10)

	ax = plt.subplot(gs[1,0])
	imgplot = ax.imshow(X_train_aug1[i].reshape((28,28)), cmap="binary")
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_title('Flip') 
	ax.title.set_size(10)

	ax = plt.subplot(gs[1,1])
	imgplot = ax.imshow(X_train_aug2[i].reshape((28,28)), cmap="binary")
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_title('Blur') 
	ax.title.set_size(10)

	ax = plt.subplot(gs[1,2])
	imgplot = ax.imshow(X_train_aug3[i].reshape((28,28)), cmap="binary")
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_title('Emboss')
	ax.title.set_size(10)

	ax = plt.subplot(gs[2,0])
	imgplot = ax.imshow(X_train_aug4[i].reshape((28,28)), cmap="binary")
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_title('Sharpen') 
	ax.title.set_size(10)

	ax = plt.subplot(gs[2,1])
	imgplot = ax.imshow(X_train_aug5[i].reshape((28,28)), cmap="binary")
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_title('Additive Gaussian Noise') 
	ax.title.set_size(10)

	ax = plt.subplot(gs[2,2])
	imgplot = ax.imshow(X_train_aug6[i].reshape((28,28)), cmap="binary")
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_title('ContrastNormalization') 
	ax.title.set_size(10)	

	plt.show()














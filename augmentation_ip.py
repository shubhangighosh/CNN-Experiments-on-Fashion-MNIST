#IMPORT MODULES
import tensorflow as tf
import pandas as pd
import numpy as np
import inception_preprocessing



#PLACEHOLDER VARIABLES
image = tf.placeholder(tf.uint8, shape=(28, 28, 1))



#LOAD ALL DATA
#Load training data
train = pd.read_csv('train.csv')
X_train = (train.ix[:,1:-1].values).astype('float32')
X_train = np.reshape(X_train,(-1,28,28, 1))[:20]

labels_train = train.ix[:,-1].values.astype('int32')
labels_train = np.reshape(labels_train, (-1,1))[:20]

augment = inception_preprocessing.preprocess_image(image, height=28, width=28, is_training=True)



#AUGMENT EACH IMAGE
with tf.Session() as sess:
	for i in range(1):
		x = 0
		for each_image in X_train:
			print(x)
			x += 1
			aug_image = sess.run(augment, feed_dict = {image :each_image})
			X_train = np.vstack((X_train, np.reshape(aug_image, (1, 28, 28, 1))))
		labels_train = np.vstack((labels_train, labels_train))

X_train = np.reshape(X_train, (-1, 784))

aug_data = np.hstack((X_train, labels_train))
np.savetxt("aug_train2.csv", aug_data, delimiter=",")


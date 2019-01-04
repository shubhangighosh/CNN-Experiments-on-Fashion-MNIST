from imgaug import augmenters as iaa
import pandas as pd, numpy as np


#Load validation data
train = pd.read_csv('train.csv').astype('float32')
X_train = (train.ix[:,1:-1].values)
labels_train = train.ix[:,-1].values
X_train= X_train.reshape(-1, 28, 28, 1)
labels_train = np.reshape(labels_train, (-1, 1))



# X_train_aug1 = iaa.Fliplr(1).augment_images(X_train)
#X_train_aug2 = iaa.AverageBlur(k=(3, 4)).augment_images(X_train)
#X_train_aug3 = iaa.Emboss(alpha=(0.8, 1.0), strength=(0, 3.0)).augment_images(X_train)
#X_train_aug4 = iaa.Sharpen(alpha=(0.8, 0.9), lightness=(0.75, 1.5)).augment_images(X_train)
#X_train_aug5 = iaa.AdditiveGaussianNoise(loc=0, scale=(5.0, 0.05*255)).augment_images(X_train)
X_train_aug6 = iaa.ContrastNormalization((2.0, 2.5)).augment_images(X_train)

# X_train_aug1 = np.reshape(X_train_aug1, (-1, 784))
#X_train_aug2 = np.reshape(X_train_aug2, (-1, 784))
#X_train_aug3 = np.reshape(X_train_aug3, (-1, 784))
#X_train_aug4 = np.reshape(X_train_aug4, (-1, 784))
#X_train_aug5 = np.reshape(X_train_aug5, (-1, 784))
X_train_aug6 = np.reshape(X_train_aug6, (-1, 784))

# aug_data1 = np.hstack((X_train_aug1, labels_train))
#aug_data2 = np.hstack((X_train_aug2, labels_train))
#aug_data3 = np.hstack((X_train_aug3, labels_train))
#aug_data4 = np.hstack((X_train_aug4, labels_train))
#aug_data5 = np.hstack((X_train_aug5, labels_train))
aug_data6 = np.hstack((X_train_aug6, labels_train))

# aug_data = np.concatenate((aug_data1, aug_data2, aug_data5, aug_data6), axis = 0)


print(aug_data6.shape)
pd.DataFrame(aug_data6).to_csv("aug_train_ia_6.csv")




# train_aug = pd.read_csv('aug_train_ia.csv')
# X_train_aug = (train_aug.ix[:,:-1].values).astype('float32')
# labels_train_aug = train_aug.ix[:,-1].values.astype('int32')

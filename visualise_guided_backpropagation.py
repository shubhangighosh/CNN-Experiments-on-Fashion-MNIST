import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
import numpy as np

#labeled_images = pd.read_csv('gb.csv')
#images = labeled_images.iloc[:,1:-1]
images = pd.read_csv('gb.csv')
print np.shape(images)
print images
#Visualize 20th image
img=images.as_matrix().reshape((10,28,28))
plt.imshow(img[0],cmap='binary')
plt.show()

plt.imshow(img[1],cmap='binary')
plt.show()

plt.imshow(img[2],cmap='binary')
plt.show()
plt.imshow(img[3],cmap='binary')
plt.show()

plt.imshow(img[4],cmap='binary')
plt.show()

plt.imshow(img[5],cmap='binary')
plt.show()

plt.imshow(img[6],cmap='binary')
plt.show()

plt.imshow(img[7],cmap='binary')
plt.show()

plt.imshow(img[8],cmap='binary')
plt.show()

plt.imshow(img[9],cmap='binary')
plt.show()

# img=images.iloc[2].as_matrix().reshape((28,28))
# plt.imshow(img,cmap='binary')
# plt.show()

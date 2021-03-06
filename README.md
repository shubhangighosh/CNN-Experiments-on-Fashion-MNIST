# CNN-Experiments-on-Fashion-MNIST  
Convolutional Neural networks have been trained on the Fashion MNIST dataset and various experiments have been run.  
The implementation was done using Tensorflow. No higher level API were used.  

Various experiments have been run:  
1. Data Augmentation  
2. Guided Backpropagation
3. Fooling the network  
4. Experimenting with Xavier/He initialisation
5. Early stopping  
6. Batch Normalisation  

The results of the above experiments can be found [here](https://github.com/shubhangighosh/CNN-Experiments-on-Fashion-MNIST/blob/master/Report.pdf).

##############################################################################  
USAGE  
##############################################################################  
#  
1.   
To run the basic model described in the problem statement :   
"python train.py"  
#  
2.  
To run the best performing model described in the report :   
"python train_best_model.py"  
OR  
"./run.sh"  
(This can only be run with augmented data, that is, run Step 4 first.)  
#  
3.  
To run either model with optional custom parameters :  
python train.py/train_best_model.py -lr <learing_rate> -batch_size <mini_batch_size> -init <0/1/2> -save_dir <directory_to_save_model>   
#  
4.  
To produce augmented data using imgaug :  
"python augmentation_ia.py"  
#  
5.
To visualize guided backprop :   
"python guided_backprop.py && python visualize_guided.py"  
#  
6.  
To visualize training and validation loss plots :   
"python train.py/train_best_model.py && python visualize_losses.py"  
#  
7.  
To visualize data - original and augmented :   
"python visualize_data.py"  
#  
8.  
To visualize conv layer 1 filters :   
"python train.py/train_best_model.py && python visualize_filters.py"  
#  
9.  
To visualize fooling :   
"python train.py/train_best_model.py && python visualize_fooling.py"  
10.  
Guided Backpropagation :   
"python guided_backprop.py && python visualize_guided.py"  







##############################################################################
DESCRIPTION OF FILES 
############################################################################## 
#
train.csv - Training data
val.csv - Validation data
test.csv - Test data
#
#
#
train.py - Contains the basic model asked for in the problem statement, not the best performing model. Can be run as 'python train.py'
#
train_best_model.py - Contains the best model which gave the highest scoreo on kaggle. Can be run as 'python train_best_model.py'
#
#
#
augmentation_ip.py - Inception Preprocessing based augmentation. Running this produces augmented data in the file "aug_train_ip.csv".
#
inception_preprocessing.py - The Tensorlfow module required to tun the above module, downloaded from "https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py"
#
augmentation_ia.py - Python library imgaug based augmentation. Running this produces augmented data in the file "aug_train_ia.csv".
#
#
#
visualize_data.py - Visualize the 784 pixels as a black and white image, for both original as well as augmented da
visualize_filters.py - Visualize the layer 1 convolutional filters as stored in 'tmp/conv1_filters.pkl'
visualize_losses.py - Generate plots of (training/validation)loss vs epochs as stored in 'tmp/losses'
visualize_fooling.py - Generate plot of pixels vs accuracy as stored in 'tmp/fooling'
visualize_guided.py - Generates visualization of guided backpropagation using gb.csv
#
#
#
guided_backprop.py - Implements guided backpropagation
gb.csv - Gradients of 10 neurons in the final convolutional layer withr respect to inputs
#
#
#
Kaggle_subs - Folder contains all the kaggle submissions numbered in chronological order(Highest number is the latest submission)







#IMPORT MODULES
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
import pandas as pd
import numpy as np
import argparse, pickle

tf.set_random_seed(1234)
#PARAMETERS
learning_rate = 0.001
epochs = 10
mini_batch_size = 50
init_method = 2		#0 for random, 1 for Xa, 2 for He 
save_dir = "tmp"  	#Directory to save pickled model in
k = 10  			#Number of classes
early_stopping = False
anneal = False


#PLACEHOLDER VARIABLES
X_ph = tf.placeholder(tf.float32, [None, 784])
Y_ph= tf.placeholder(tf.float32, [None, k])
training_phase = tf.placeholder(tf.bool)
lr = tf.placeholder(tf.float32)
drops = tf.placeholder(tf.float32, [1, None])

#Reshape input
shaper = tf.reshape(X_ph, [-1, 28, 28, 1])
	
#NEURAL NETWORK
def CNN():
	
	if init_method == 0:
		initializer_wb= tf.random_normal_initializer()
	elif init_method == 1:
		initializer_wb= tf.contrib.layers.xavier_initializer()
	elif init_method == 2:
		initializer_wb= tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")		

	conv_layer_1 = tf.layers.conv2d( inputs=shaper, 
									 filters=32,
									 kernel_size=[3, 3],
									 padding="same",
									 strides=1,
									 activation=tf.nn.relu,
									 kernel_initializer=initializer_wb, 
									 name = "conv1")
	# pool_layer_1 = tf.layers.max_pooling2d(inputs=conv_layer_1, pool_size=[2, 2], strides=2)
	# pool_layer_1_dropped = tf.nn.dropout(pool_layer_1, drops[0])
	# batched_1 = tf.contrib.layers.batch_norm(	conv_layer_1,center=True, scale=True, 
 #                                          		is_training = training_phase)


  	conv_layer_2 = tf.layers.conv2d( inputs=conv_layer_1,
									  filters=64,
									  kernel_size=[3, 3],
									  padding="same",
									  activation=tf.nn.relu)
	pool_layer_2 = tf.layers.max_pooling2d(inputs=conv_layer_2, pool_size=[2, 2], strides=2)
	#pool_layer_2_dropped = tf.nn.dropout(pool_layer_2, drops[0,0])
	# batched_2 = tf.contrib.layers.batch_norm(	conv_layer_2,center=True, scale=True, 
 #                                          		is_training = training_phase)



	conv_layer_3 = tf.layers.conv2d( inputs=pool_layer_2,
									  filters=128,
									  kernel_size=[3, 3],
									  padding="same",
									  activation=tf.nn.relu)
	


	conv_layer_4 = tf.layers.conv2d( inputs=conv_layer_3,
							  filters=256,
							  kernel_size=[3, 3],
							  padding="same",
							  activation=tf.nn.relu)
	pool_layer_4 = tf.layers.max_pooling2d(inputs=conv_layer_4, pool_size=[2, 2], strides=2)
	# pool_layer_3_dropped = tf.nn.dropout(pool_layer_3, drops[2])
	# batched_3 = tf.contrib.layers.batch_norm(	pool_layer_3,center=True, scale=True, 
 #                                          		is_training = training_phase)
	#batched_3_dropped =  tf.nn.dropout(pool_layer_3, drops[0,0])


	# conv_layer_4 = tf.layers.conv2d( inputs=batched_3_dropped,
	# 						  filters=128,
	# 						  kernel_size=[3, 3],
	# 						  padding="same",
	# 						  activation=tf.nn.relu)
	# batched_4 = tf.contrib.layers.batch_norm(	conv_layer_4,center=True, scale=True, 
 #                                          		is_training = training_phase)
	
	# conv_layer_5 = tf.layers.conv2d( inputs=pool_layer_4,
	# 						  filters=256,
	# 						  kernel_size=[2, 2],
	# 						  padding="same",
	# 						  activation=tf.nn.relu)
	# pool_layer_5 = tf.layers.max_pooling2d(inputs=conv_layer_5, pool_size=[2, 2], strides=2)
	# # batched_5 = tf.contrib.layers.batch_norm(	pool_layer_5,center=True, scale=True, 
 # #                                          		is_training = training_phase)
	# batched_5_dropped =  tf.nn.dropout(pool_layer_5, drops[0,1])



	flattened_layer = tf.reshape(pool_layer_4, [-1, 7 * 7 * 256])
	# batched_flat = tf.contrib.layers.batch_norm(flattened_layer,center=True, scale=True, is_training = training_phase)


	dense_layer_1 = tf.layers.dense(inputs=flattened_layer, units=1000, activation=tf.nn.relu)
	#batched_dense_layer_1 = tf.contrib.layers.batch_norm(dense_layer_1,center=True, scale=True, is_training = training_phase)
	
	
	dense_layer_2 = tf.layers.dense(inputs=dense_layer_1, units=1000, activation=tf.nn.relu)
	#batched_dense_layer_2 = tf.contrib.layers.batch_norm(dense_layer_2,center=True, scale=True, is_training = phase)
	
	# dense_layer_2_dropped = tf.nn.dropout(dense_layer_2, drops[0,0])


	dense_layer_3 = tf.layers.dense(inputs=dense_layer_2, units=1000, activation=tf.nn.relu)

	logit_layer = tf.layers.dense(inputs=dense_layer_3, units=k)
	# output = tf.nn.softmax(logits, name="softmax_tensor")
	# classes = tf.argmax(input=output, axis=1)
	return logit_layer






#INITIALIZE WEIGHTS AND BIASES - currently not being used anywhere
# W = {
# 	'c1': tf.Variable(tf.random_normal([3, 3, 1, 64])),
# 	'c2': tf.Variable(tf.random_normal([3, 3, 64, 128])),
# 	'c3': tf.Variable(tf.random_normal([3, 3, 64, 256])),
# 	'c4': tf.Variable(tf.random_normal([3, 3, 64, 256])),

# 	'f1': tf.Variable(tf.random_normal([3 * 3 * 256, 1024])),
# 	'f2': tf.Variable(tf.random_normal([1024, 1024])),

# 	'out': tf.Variable(tf.random_normal([1024, k]))
# 	}

# b = { 	
# 	'c1': tf.Variable(tf.random_normal([64])),
# 	'c2': tf.Variable(tf.random_normal([128])),
# 	'c3': tf.Variable(tf.random_normal([256])),
# 	'c4': tf.Variable(tf.random_normal([256])),

# 	'f1': tf.Variable(tf.random_normal([1024])),
# 	'f2': tf.Variable(tf.random_normal([1024])),

# 	'out': tf.Variable(tf.random_normal([k]))
# 	}














#PARSE ARGUMENTS
parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--lr', action="store", dest = 'lr', type=float)
parser.add_argument('--batch_size', action="store", dest="batch_size", type=int)
parser.add_argument('--init', action="store", dest="init", type=int)
parser.add_argument('--save_dir', action="store", dest="save_dir")
args = parser.parse_args()
print(args)
if(args.lr):
	learning_rate = args.lr
if(args.batch_size):
	if(not(mini_batch_size == 1 or mini_batch_size%5 == 0)):
		raise ValueError('Valid values for batch_size are 1 and multiples of 5 only')
	else:
		mini_batch_size = args.batch_size	
if(args.init):
	init_method = args.init
if(args.save_dir):
	save_dir = args.save_dir





#LOAD ALL DATA
#Load training data
train = pd.read_csv('train.csv')
X_train = (train.ix[:,1:-1].values).astype('float32')
labels_train = train.ix[:,-1].values.astype('int32')
#
#Load augmented data
train_aug = pd.read_csv('aug_train_ia.csv')
X_train_aug = (train_aug.ix[:,1:-1].values).astype('float32')
#
#Combine original and augmented
labels_train_aug = train_aug.ix[:,-1].values.astype('int32')
labels_train = np.reshape(labels_train, (-1,1))
labels_train_aug = np.reshape(labels_train_aug, (-1,1))
X_train = np.concatenate((X_train, X_train_aug), axis=0)
labels_train = np.concatenate((labels_train, labels_train_aug), axis=0)
labels_train = np.reshape(labels_train, (-1))
#
#Final data shape
print(X_train.shape, labels_train.shape)
#
#Convert to one-hot
Y_train = np.zeros((labels_train.shape[0], 10))
Y_train[np.arange(labels_train.shape[0]), labels_train] = 1
#
#Standardise
mean_X = X_train.mean().astype(np.float32)
# std_X = X_train.std().astype(np.float32)
X_train = (X_train - mean_X)/255

#Load validation data
val = pd.read_csv('val.csv')
X_val = (val.ix[:,1:-1].values).astype('float32')
labels_val = val.ix[:,-1].values.astype('int32')
#
#Convert to one-hot
Y_val = np.zeros((labels_val.shape[0], 10))
Y_val[np.arange(labels_val.shape[0]), labels_val] = 1
#
#Standardise
# mean_X = X_val.mean().astype(np.float32)
# std_X = X_val.std().astype(np.float32)
X_val = (X_val - mean_X)/255

#Load test data
test = pd.read_csv("test.csv")
X_test = (test.ix[:,1:].values).astype('float32')
test_ids = (test.ix[:,0].values).astype('int32')
#
#Standardise
# mean_X = X_test.mean().astype(np.float32)
# std_X = X_test.std().astype(np.float32)
X_test = (X_test - mean_X)/255





















#BUILD MODEL
#Build optimizer
last_layer = CNN()
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=last_layer, labels=Y_ph))
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)


# Evaluate model
matching_pred = tf.equal(tf.argmax(last_layer, 1), tf.argmax(Y_ph, 1))
accuracy = tf.reduce_mean(tf.cast(matching_pred, tf.float32))

#Shuffler
batch_shuffle = tf.train.shuffle_batch([X_train, Y_train], enqueue_many=True, batch_size=mini_batch_size, capacity=X_train.shape[0], min_after_dequeue=mini_batch_size, allow_smaller_final_batch=True)
#Log Summarizer
# tf.summary.scalar("cost", cost)
# tf.summary.scalar("accuracy", accuracy)
# summary_op = tf.summary.merge_all()

# Comment out for complete training
# X_train = X_train[:30]
# Y_train = Y_train[:30]
# X_val = X_val[:30]
# Y_val = Y_val[:30]
# X_test = X_test[:30]
# test_ids = test_ids[:30]


#RUN MODEL
initializer = tf.global_variables_initializer()
saver = tf.train.Saver()
# train_writer = tf.summary.FileWriter("tflogs/train/", graph=tf.get_default_graph())
# val_writer = tf.summary.FileWriter("tflogs/val/", graph=tf.get_default_graph())
last_five_val_losses = np.zeros(5)
prev_loss = np.inf
loss_file = open("tmp/losses", 'w')

with tf.Session() as sess:
	# X_train = sess.run()
	sess.run([initializer, shaper], feed_dict={X_ph : X_train})
	step = 0
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)		

	for e in range(epochs):	
		total_loss_train = 0	
		total_acc = 0
		num_acc = 0
		print("\nEpoch " + str(e+1))

		for i in range(0, X_train.shape[0], mini_batch_size):
			#print(step)
			step += 1
			X_train_mini, Y_train_mini = sess.run(batch_shuffle)			
			sess.run([optimizer], feed_dict={X_ph: X_train_mini, Y_ph: Y_train_mini, training_phase: True, lr:learning_rate, drops: [[0.4]]})
			#Uncomment for Tensorboard Summary
			# _, summary = sess.run([optimizer, summary_op], feed_dict={X_ph: X_train_mini, Y_ph: Y_train_mini, training_phase: True})
			# train_writer.add_summary(summary, step)
			# summary = sess.run(summary_op, feed_dict={X_ph: X_val, Y_ph: Y_val, training_phase: False})			
			# val_writer.add_summary(summary, step)	

			loss_train, acc = sess.run([cost, accuracy], feed_dict={X_ph: X_train_mini, Y_ph: Y_train_mini, training_phase: False, drops: [[0.4]]})
			total_loss_train += loss_train
			total_acc += acc
			num_acc += 1

		avg_acc = total_acc/num_acc
		print("Training :  Loss= {:.6f}".format(total_loss_train) + ", Accuracy= " + "{:.5f}".format(avg_acc))
		#Validation loss after every epoch
		loss_val, acc_val = sess.run([cost, accuracy], feed_dict={X_ph: X_val, Y_ph: Y_val, training_phase: False, drops: [[0.4]]})
		print("Validation : Loss= {:.6f}".format(loss_val) + ", Accuracy= " + "{:.5f}".format(acc_val))
		loss_file.write(str(e+1)+" "+str(loss_train)+" "+str(loss_val))

		if (anneal == True) or (early_stopping == True):
			save_path = saver.save(sess, save_dir+"/"+str(e)+".ckpt")
		if(anneal == True):
			if(loss_val > prev_loss):
				learning_rate = learning_rate/2.0
				saver.restore(sess, save_dir+"/"+str(max(0,e-1))+".ckpt")
			else:
				prev_loss = loss_val	
		if(early_stopping == True):
			np.roll(last_five_val_losses, -1)
			last_five_val_losses[-1] = loss_val
			if loss_val > last_five_val_losses[0]:
				saver.restore(sess, save_dir+"/"+str(max(0,e-5))+".ckpt")
				break

	coord.request_stop()
	coord.join(threads)
	loss_file.close()



	#If testing required, get predictions
	predictions = sess.run(last_layer, feed_dict={X_ph: X_test, training_phase: False, drops : np.ones((1, 5))})
	predictions = np.argmax(predictions, axis=1)		
	#WRITE PREDICTIONS FOR SUBMISSION
	submissions=pd.DataFrame({"id": test_ids, "label": predictions})
	submissions.to_csv("test_submission.csv", index=False, header=True)

	













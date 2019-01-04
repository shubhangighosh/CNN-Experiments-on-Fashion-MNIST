#IMPORT MODULES
import tensorflow as tf
import pandas as pd
import numpy as np
import argparse, pickle, sys
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

tf.set_random_seed(1234)
#PARAMETERS
learning_rate = 0.001
epochs = 5
mini_batch_size = 5
init_method = 2 	#0 for random, 1 for Xa, 2 for He 
save_dir = "tmp"  	#Directory to save pickled model in
k = 10  			#Number of classes
early_stopping = False
anneal = False


#PLACEHOLDER VARIABLES
X_ph = tf.placeholder(tf.float32, [None, 784])
Y_ph= tf.placeholder(tf.float32, [None, k])
training_phase = tf.placeholder(tf.bool)
lr = tf.placeholder(tf.float32)


shaper = tf.reshape(X_ph, [-1, 28, 28, 1])






@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    return tf.where(0. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros_like(grad))



#NEURAL NETWORK
def CNN():

	# g = tf.get_default_graph()
	# with g.gradient_override_map({'Relu': 'GuidedRelu'}):
	
	if init_method == 0:
		initializer_wb= tf.random_normal_initializer()
	elif init_method == 1:
		initializer_wb= tf.contrib.layers.xavier_initializer()
	elif init_method == 2:
		initializer_wb= tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")		

	conv_layer_1 = tf.layers.conv2d( inputs=shaper, 
									 filters=64,
									 kernel_size=[3, 3],
									 padding="same",
									 strides=1,
									 activation=tf.nn.relu,
									 kernel_initializer=initializer_wb,
									 name = "conv1")
	pool_layer_1 = tf.layers.max_pooling2d(inputs=conv_layer_1, pool_size=[2, 2], strides=2)




  	conv_layer_2 = tf.layers.conv2d( inputs=pool_layer_1,
									  filters=128,
									  kernel_size=[3, 3],
									  padding="same",
									  activation=tf.nn.relu,
									  kernel_initializer=initializer_wb)
	pool_layer_2 = tf.layers.max_pooling2d(inputs=conv_layer_2, pool_size=[2, 2], strides=2)




	conv_layer_3 = tf.layers.conv2d( inputs=pool_layer_2,
									  filters=256,
									  kernel_size=[3, 3],
									  padding="same",
									  activation=tf.nn.relu,
									 kernel_initializer=initializer_wb)
	


	conv_layer_4 = tf.layers.conv2d( inputs=conv_layer_3,
							  filters=256,
							  kernel_size=[3, 3],
							  padding="same",
							  activation=tf.nn.relu,
							  kernel_initializer=initializer_wb)
	pool_layer_3 = tf.layers.max_pooling2d(inputs=conv_layer_4, pool_size=[2, 2], strides=2)

	grads_40 = [tf.gradients(conv_layer_4[0,i,0,:], X_ph) for i in np.arange(5)]
	grads_41 = [tf.gradients(conv_layer_4[0,i,1,:], X_ph) for i in np.arange(5)]

	flattened_layer = tf.reshape(pool_layer_3, [-1, 3 * 3 * 256])


	dense_layer_1 = tf.layers.dense(inputs=flattened_layer, units=1024, activation=tf.nn.relu,kernel_initializer=initializer_wb)
	dense_layer_2 = tf.layers.dense(inputs=dense_layer_1, units=1024, activation=tf.nn.relu,kernel_initializer=initializer_wb)



	#dropped = tf.nn.dropout(dense_layer_2, dropout)
	#tf.contrib.layers.batch_norm(dropped,center=True, scale=True, is_training = training_phase,)


	logit_layer = tf.layers.dense(inputs=dense_layer_2, units=k)
# output = tf.nn.softmax(logits, name="softmax_tensor")
# classes = tf.argmax(input=output, axis=1)
	return logit_layer, grads_40, grads_41







# #INITIALIZE WEIGHTS AND BIASES - currently not being used anywhere
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
#Convert to one-hot
Y_train = np.zeros((labels_train.shape[0], 10))
Y_train[np.arange(labels_train.shape[0]), labels_train] = 1
#
#Standardise
mean_X = X_train.mean().astype(np.float32)
# std_X = X_train.std().astype(np.float32)
X_train = (X_train - mean_X)/255
#
#Final data shape
print(X_train.shape, labels_train.shape)

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
mean_X = X_val.mean().astype(np.float32)
# std_X = X_val.std().astype(np.float32)
X_val = (X_val - mean_X)/255
# X_val = (X_val/255-0.5)*2

#Load test data
test = pd.read_csv("test.csv")
X_test = (test.ix[:,1:].values).astype('float32')
test_ids = (test.ix[:,0].values).astype('int32')
#
#Standardise
mean_X = X_test.mean().astype(np.float32)
# std_X = X_test.std().astype(np.float32)
# X_test = (X_test/255-0.5)*2
X_test = (X_test - mean_X)/255






#BUILD MODEL
#Build optimizer
# g = tf.get_default_graph()
# with g.gradient_override_map({'Relu': 'GuidedRelu'}):

last_layer, grads_40, grads_41 = CNN()
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=last_layer, labels=Y_ph))
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

# Evaluate model
matching_pred = tf.equal(tf.argmax(last_layer, 1), tf.argmax(Y_ph, 1))
accuracy = tf.reduce_mean(tf.cast(matching_pred, tf.float32))

#Shuffler
batch_shuffle = tf.train.shuffle_batch([X_train, Y_train], enqueue_many=True, batch_size=mini_batch_size, capacity=mini_batch_size+1, min_after_dequeue=mini_batch_size, allow_smaller_final_batch=True)

#Log Summarizer
# tf.summary.scalar("cost", cost)
# tf.summary.scalar("accuracy", accuracy)
# summary_op = tf.summary.merge_all()



#Comment out for complete training
X_train = X_train[:-1]
Y_train = Y_train[:-1]
X_val = X_val[:5]
Y_val = Y_val[:5]
X_test = X_test[:5]
test_ids = test_ids[:5]



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
	sess.run(initializer)
	step = 0
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)	


	# g = tf.get_default_graph()

	# with g.gradient_override_map({'Relu': 'GuidedRelu'}):	

	for e in range(epochs):		
		total_loss_train = 0	
		total_acc = 0
		num_acc = 0
		print("\nEpoch " + str(e+1))

		for i in range(0, X_train.shape[0], mini_batch_size):
		#print(step)
			step += 1
			X_train_mini, Y_train_mini = sess.run(batch_shuffle)			
			sess.run([optimizer], feed_dict={X_ph: X_train_mini, Y_ph: Y_train_mini, training_phase: True, lr:learning_rate})
			#Uncomment for Tensorboard Summary
			# _, summary = sess.run([optimizer, summary_op], feed_dict={X_ph: X_train_mini, Y_ph: Y_train_mini, training_phase: True})
			# train_writer.add_summary(summary, step)
			# summary = sess.run(summary_op, feed_dict={X_ph: X_val, Y_ph: Y_val, training_phase: False})			
			# val_writer.add_summary(summary, step)	

			loss_train, acc = sess.run([cost, accuracy], feed_dict={X_ph: X_train_mini, Y_ph: Y_train_mini, training_phase: False})
			total_loss_train += loss_train
			total_acc += acc
			num_acc += 1

		avg_acc = total_acc/num_acc
		print("Training :  Loss= {:.6f}".format(total_loss_train) + ", Accuracy= " + "{:.5f}".format(avg_acc))
		#Validation loss after every epoch
		loss_val, acc_val = sess.run([cost, accuracy], feed_dict={X_ph: X_val, Y_ph: Y_val, training_phase: False})
		print("Validation : Loss= {:.6f}".format(loss_val) + ", Accuracy= " + "{:.5f}".format(acc_val))
		loss_file.write(str(e+1)+" "+str(total_loss_train)+" "+str(loss_val)+"\n")

		if (anneal == True) or (early_stopping == True):
			save_path = saver.save(sess, save_dir+"/"+str(e+1)+".ckpt")
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
		#loss_file.close()
	g = tf.get_default_graph()

	with g.gradient_override_map({'Relu': 'GuidedRelu'}):		
		grad_gb0 = sess.run(grads_40, feed_dict={X_ph: X_val[:1], Y_ph: Y_val[:1], training_phase: False})
		grad_gb1 = sess.run(grads_41, feed_dict={X_ph: X_val[:1], Y_ph: Y_val[:1], training_phase: False})
		print np.shape(grad_gb0), np.shape(grad_gb1), np.shape(grads_40[1][0][0])
		gb_arr = []
		for i in np.arange(5):
			gb_arr.append(abs(np.array(grad_gb0[i])*(10**6)).reshape(784))
		for i in np.arange(5):
			gb_arr.append(abs(np.array(grad_gb1[i])*(10**6)).reshape(784))	
		gb_arr =  np.array(gb_arr)
		
		df = pd.DataFrame(gb_arr)
		df.to_csv("gb.csv", index=False, header=True)

		# print(sess.run(grads_4, feed_dict={X_ph: X_val[:1], Y_ph: Y_val[:1], training_phase: True})
		# 		)
	












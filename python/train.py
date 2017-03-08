import numpy as np 
import tensorflow as tf
from PIL import Image
import os

def getImages():
	#load images to np arrays
	path = "../processed-data/"
	files = os.listdir(path)
	imgs = np.zeros((40,40,3,len(files)))
	for i in range(len(files)):
		imgs[:,:,:,i] = Image.open(path + files)
	return imgs


imgs = getImages
labels = np.load("labels.npy")




train = tf.constant(imgs)
labels = tf.constant(labels)
filter_layer1 = tf.Variable(tf.random_normal([3,3,3,3]))
filter_layer2 = tf.Variable(tf.random_normal([2,2,3,3]))

def model(data):
	conv1 = tf.nn.conv2d(data, filter_layer1, [1,2,2,1], padding='same')
	hidden = tf.nn.relu(conv1 + layer1_biases)
	conv2 = tf.nn.conv2d(conv1, filter_layer2, [1,2,2,1])
	return tf.nn.relu(conv2 + layer2_biases)

logits = model(train)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))
optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

train_prediction = tf.nn.softmax(logits)

num_steps = 1001
with tf.Session(graph=graph) as Session:
	tf.global_variables_initializer().run()
	print('Initialized')
	for step in range(num_steps):
		#do the classify


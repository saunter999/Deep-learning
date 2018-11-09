#!/usr/bin/env python
##from __future__ import print_function
from scipy import *
from keras.datasets import mnist
from keras import models
from keras import layers
import tensorflow as tf
#from keras.utils import to_categorical

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import cm


def vectorize_labels(labels,num_class=10):
	tmp=zeros((len(labels),num_class))
	for i in range(len(labels)):
	    tmp[i,labels[i]]=1.
	return tmp


if __name__=="__main__":
	(train_images,train_labels),(test_images,test_labels) = mnist.load_data()
#	print  type(train_images),type(train_labels) 
#	print train_images.dtype
#	print  train_images.shape,train_labels.shape 
#	print  type(test_images),type(test_labels) 
#	print  test_images.shape,test_labels.shape 
#	for i in range(len(test_labels)):
#	    print test_labels[i]
#	print (train_images[0].max())

#	plt.imshow(train_images[4],cmap=None)
#	plt.imshow(train_images[4],cmap=cm.binary)
#	plt.show();exit()

	print  "---------Building network-----------" 
	network=models.Sequential()
	network.add(layers.Dense(512,activation='relu',input_shape=(28*28,))) ## hidden layer with 512 neurons
	network.add(layers.Dense(10,activation='softmax'))                    ## output layer with 10 neurons,which corresponds to 10 digits
	network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

	print  "---------Preparing input layer-----------" 
	train_images=train_images.reshape((60000,28*28))
	train_images=train_images.astype('float32')/255
	test_images=test_images.reshape((10000,28*28))
	test_images=test_images.astype('float32')/255

	print  "---------Preparing labels-----------" 
	##convert labels to a vector for each object in the sample
	train_labels=vectorize_labels(train_labels)
	test_labels=vectorize_labels(test_labels)
	
	
	print "----------Learning/training the model-----"
	network.fit(train_images,train_labels,epochs=5,batch_size=128)

	print "----------testing the network against the test_images----"
	test_loss,test_acc=network.evaluate(test_images,test_labels)
	print "test_loss:",test_loss
	print "test_acc:",test_acc

	plt.show()

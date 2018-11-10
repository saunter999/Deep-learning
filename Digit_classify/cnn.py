#!/usr/bin/env python
from keras import models,layers
from keras.datasets import mnist
from keras.utils import to_categorical

def Compute_parameters(n,m,k,l):
	"""
	How to compute the number of parameters in each cnn layer:
	Definition n--width of filter
		   m--height of filter
		   k--number of input feature maps 
		   l--number of output feature maps
	Then number of paramters  #= (n*m*k+1)*l
	"""
	print "# of paramteres in this layer=",(n*m*k+1)*l



if __name__=='__main__':
	model=models.Sequential()
	model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))  ##32 is the depth of the kernal,(3,3) is the filter/kernal shape
	Compute_parameters(3,3,1,32)
	model.add(layers.MaxPooling2D(2,2))
	model.add(layers.Conv2D(64,(3,3),activation='relu'))
	Compute_parameters(3,3,32,64)
	model.add(layers.MaxPooling2D(2,2))
	model.add(layers.Conv2D(64,(3,3),activation='relu'))
	Compute_parameters(3,3,64,64)
	model.add(layers.Flatten())
	model.add(layers.Dense(64,activation='relu'))
	model.add(layers.Dense(10,activation='softmax'))
	print model.summary()
	(train_images,train_labels),(test_images,test_labels) = mnist.load_data()
	train_images=train_images.reshape((60000,28,28,1))
	train_images=train_images.astype('float32')/255
	test_images=test_images.reshape((10000,28,28,1))
	test_images=test_images.astype('float32')/255
	train_labels=to_categorical(train_labels)
	test_labels=to_categorical(test_labels)
	model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
	model.fit(train_images,train_labels,epochs=5,batch_size=64)
	test_loss,test_acc=model.evaluate(test_images,test_labels)
	print 'test_acc=',test_acc

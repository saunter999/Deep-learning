#!/usr/bin/env python

from keras import models,layers
from keras.datasets import imdb
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt




def vectorize_sequences(sequences,dimension=10000):
	results=np.zeros((len(sequences),dimension))
#	print len(sequences)
	for i,sequence in enumerate(sequences):
	    #print i,type(sequence),sequence
	    results[i,sequence]=1.
	return results
		



if __name__=="__main__":
        ## Downloading data and prepare input layer
	(train_data,train_labels),(test_data,test_labels)=imdb.load_data(num_words=10000)
#	print type(train_data)
	print train_data.ndim,train_data.shape
#	print train_data[0]
#	print train_data[1]
#	print len(train_data[0]),type(train_data[0])

	x_train=vectorize_sequences(train_data)
	x_test=vectorize_sequences(test_data)
	y_train=np.asarray(train_labels,dtype='float32')
	y_test=np.asarray(test_labels,dtype='float32')
	print "Downloading done"

	##Building the structure of the network
	model=models.Sequential()
	model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
	model.add(layers.Dense(16,activation='relu'))
	model.add(layers.Dense(1,activation='sigmoid'))
	model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

	##Training the network
	print "starting training the network"
	x_val=x_train[:10000]
	y_val=y_train[:10000]
	partial_x_train=x_train[10000:]
	partial_y_train=y_train[10000:]
	history=model.fit(partial_x_train,partial_y_train,epochs=20,batch_size=512,validation_data=(x_val,y_val))

	##Plotting the output
	history_dict=history.history
	print (history_dict).keys()
	loss_values=history_dict['loss']
	val_loss_values=history_dict['val_loss']
	epochs=range(1,len(loss_values)+1)
	plt.figure(1)
	plt.plot(epochs,loss_values,label='training loss')
	plt.plot(epochs,val_loss_values,label='validation loss')
	plt.xlabel('Epochs')
	plt.ylabel('loss')
	plt.legend(loc=0)

	acc=history_dict['acc']
	val_acc=history_dict['val_acc']
	plt.figure(2)
	plt.plot(epochs,acc,label='training acc')
	plt.plot(epochs,val_acc,label='validation acc')
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.legend(loc=0)

	plt.show()

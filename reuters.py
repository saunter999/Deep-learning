#!/usr/bin/env python
from keras.datasets import reuters
import numpy as np
from keras import models,layers

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
		

def to_one_hot(labels,dimension=46):
	results=np.zeros((len(labels),dimension))
	for i,label in enumerate(labels):
	    results[i,label]=1.
	return results
		

         





if __name__=="__main__":
	(train_data,train_labels),(test_data,test_labels)=reuters.load_data(num_words=10000)
	print len(train_data),len(test_data)
	print train_labels ,max(train_labels)
	
	x_train=vectorize_sequences(train_data)
	x_test=vectorize_sequences(test_data)
	one_hot_train_labels=to_one_hot(train_labels)
	one_hot_test_labels=to_one_hot(test_labels)
#	print one_hot_train_labels[0]

	##setting up the network
	model=models.Sequential()
	model.add(layers.Dense(64,activation='relu',input_shape=(10000,)))
	model.add(layers.Dense(6,activation='relu'))
	model.add(layers.Dense(46,activation='softmax'))
	model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

	##training the network
	x_val=x_train[:1000]
	y_val=one_hot_train_labels[:1000]
	partial_x_train=x_train[1000:]
	partial_y_train=one_hot_train_labels[1000:]
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

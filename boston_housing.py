#!/usr/bin/env python

from keras.datasets import boston_housing
from keras import models,layers




if __name__=='__main__':
	(train_data,train_targets),(test_data,test_targets)=boston_housing.load_data()
#	print train_data.shape,train_targets.shape
#	print test_data.shape,test_targets.shape
	print 'train_data:',train_data[0]
#	print 'test_data:',test_data[0]
#	print 'train_targets:',train_targets[0]
#	print 'test_targets:',test_targets[0]

	##since the entry in the train_data has different meanings, we need to "normalize" each entry using the characteristics in the population.
	mean=train_data.mean(axis=0)
	train_data-=mean
	std=train_data.std(axis=0)
	train_data/=std
	test_data-=mean
	test_data/=std

	##setting up the model
	model=models.Sequential()
	model.add(layers.Dense(64,activation='relu',input_shape=(train_data.shape[1],)))
	model.add(layers.Dense(64,activation='relu'))
	model.add(layers.Dense(1))
	model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])

	##Training the network
	print "starting training the network"
	history=model.fit(train_data,train_targets,epochs=80,batch_size=16,verbose=0)
	test_mse,test_mae=model.evaluate(test_data,test_targets)
	print test_mae,test_mse
	

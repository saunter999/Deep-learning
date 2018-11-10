#!/usr/bin/env python
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator as IDG 
from keras import layers,models
from keras import optimizers
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import cm

def extract_features(dir,sample_count):
    features=np.zeros(shape=(sample_count,4,4,512))
    labels=np.zeros(sample_count)
    generator=datagen.flow_from_directory(dir,target_size=(150,150),batch_size=batch_size,class_mode='binary')
    i=0 
    for inputs_batch,labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
	features[i*batch_size:(i+1)*batch_size]=features_batch
	labels[i*batch_size:(i+1)*batch_size]=labels_batch
	i+=1
	if i*batch_size>=sample_count:
	   break
	return features,labels


if __name__=="__main__":
     ##using VGG16 to pre-extract feature used to train the FC layer
     conv_base=VGG16(weights='imagenet',include_top=False,input_shape=(150,150,3))
     #print conv_base.summary()
     train_dir='/Users/han/keras_use/examples/CNN/dogsvscats/train'
     validation_dir='/Users/han/keras_use/examples/CNN/dogsvscats/validation'
     datagen=IDG(rescale=1./255)
     batch_size=20
     train_features,train_labels =extract_features(train_dir,2000)
     validation_features,validation_labels =extract_features(validation_dir,1000)
     train_features=np.reshape(train_features,(2000,4*4*512))
     validation_features=np.reshape(validation_features,(1000,4*4*512))

     ##FC layer
     model=models.Sequential()
     model.add(layers.Dense(256,activation='relu',input_dim=4*4*512))
     model.add(layers.Dropout(0.5))
     model.add(layers.Dense(1,activation='sigmoid'))
     model.compile(optimizer=optimizers.RMSprop(lr=1e-4),loss='binary_crossentropy',metrics=['accuracy'])
     history=model.fit(train_features,train_labels,epochs=30,batch_size=20,validation_data=(validation_features,validation_labels))

     ##print/plot training result
     acc=history.history['acc']
     val_acc=history.history['val_acc']
     loss=history.history['loss']
     val_loss=history.history['val_loss']
     epochs=range(1,len(acc)+1)
     plt.figure(1)
     plt.plot(epochs,acc,'bo-',label="Training acc")
     plt.plot(epochs,val_acc,'r*-',label="Validation acc")
     plt.xlabel('epoch',size='large')
     plt.ylabel('acc',size='large')
     plt.legend()
     plt.savefig('Accuary.png')
     plt.figure(2)
     plt.plot(epochs,loss,'bo-',label="Training loss")
     plt.plot(epochs,val_loss,'r*-',label="Validation loss")
     plt.xlabel('epoch',size='large')
     plt.ylabel('loss',size='large')
     plt.legend()
     plt.savefig('Loss.png')
     plt.show()


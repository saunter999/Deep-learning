#!/usr/bin/env python
import os,shutil
from keras import layers,models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator 

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import cm

def Data_prep(train_dir,validation_dir,test_dir):
     ###creating top directories
     org_train_dataset_dir="/Users/han/keras_use/examples/Datasets/dogsvscats/train"
#     org_test_dataset_dir="/Users/han/keras_use/examples/Datasets/dogsvscats/test"
     Mkdir(train_dir)
     Mkdir(validation_dir)
     Mkdir(test_dir)
   
     ###creating sub directories for training,validation and testing separately
     train_cats_dir=os.path.join(train_dir,'cats')
     train_dogs_dir=os.path.join(train_dir,'dogs')
     validation_cats_dir=os.path.join(validation_dir,'cats')
     validation_dogs_dir=os.path.join(validation_dir,'dogs')
     test_cats_dir=os.path.join(test_dir,'cats')
     test_dogs_dir=os.path.join(test_dir,'dogs')
     Mkdir(train_cats_dir)
     Mkdir(train_dogs_dir)
     Mkdir(validation_cats_dir)
     Mkdir(validation_dogs_dir)
     Mkdir(test_cats_dir)
     Mkdir(test_dogs_dir)
     image_cp('cat',org_train_dataset_dir,train_cats_dir,0,1000)
     image_cp('dog',org_train_dataset_dir,train_dogs_dir,0,1000)
     image_cp('cat',org_train_dataset_dir,validation_cats_dir,1000,1500)
     image_cp('dog',org_train_dataset_dir,validation_dogs_dir,1000,1500)
     image_cp('cat',org_train_dataset_dir,test_cats_dir,1500,2000)
     image_cp('dog',org_train_dataset_dir,test_dogs_dir,1500,2000)

def Mkdir(dir):
    if not os.path.isdir(dir):
       os.mkdir(dir)

def image_cp(anm,src_dir,dst_dir,st,end):
    fnames=[anm+'.{}.jpg'.format(i) for i in range(st,end)]
    for fname in fnames:
        src=os.path.join(src_dir,fname)
	dst=os.path.join(dst_dir,fname)
	shutil.copyfile(src,dst)




if __name__=="__main__":
     ##Dowolading source images
     train_dir='/Users/han/keras_use/examples/CNN/dogsvscats/train'
     validation_dir='/Users/han/keras_use/examples/CNN/dogsvscats/validation'
     test_dir='/Users/han/keras_use/examples/CNN/dogsvscats/test'
     if not os.path.isdir(train_dir): 
        print "Creating train_dir..."
	Data_prep(train_dir,validation_dir,test_dir)
     else:
        print "Train_dir existed!!!"
     if not os.path.isdir(validation_dir): 
        print "Creating validation_dir..."
	Data_prep(train_dir,validation_dir,test_dir)
     else:
        print "Validation_dir existed!!!"
     if not os.path.isdir(test_dir): 
        print "Creating test_dir..."
	Data_prep(train_dir,validation_dir,test_dir)
     else:
        print "Test_dir existed!!!"
    
     ##converting image data to floats data
     train_datagen=ImageDataGenerator(rescale=1./255)
     validation_datagen=ImageDataGenerator(rescale=1./255)
     train_generator=train_datagen.flow_from_directory(train_dir,target_size=(150,150),batch_size=20,class_mode='binary')
     validation_generator=validation_datagen.flow_from_directory(validation_dir,target_size=(150,150),batch_size=20,class_mode='binary')
#     for data_batch,label_batch in validation_generator:
 #        print "data shape",data_batch.shape
  #       print "label shape",label_batch.shape
     
    
     ##Building CNN
     model=models.Sequential()
     model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3))) 
     model.add(layers.MaxPooling2D(2,2))
     model.add(layers.Conv2D(64,(3,3),activation='relu')) 
     model.add(layers.MaxPooling2D(2,2)) 
     model.add(layers.Conv2D(128,(3,3),activation='relu')) 
     model.add(layers.MaxPooling2D(2,2))
     model.add(layers.Conv2D(128,(3,3),activation='relu')) 
     model.add(layers.MaxPooling2D(2,2))
     model.add(layers.Flatten())
     model.add(layers.Dense(512,activation='relu'))
     model.add(layers.Dense(1,activation='sigmoid'))
     print model.summary()
     model.compile(optimizer=optimizers.RMSprop(lr=1e-4),loss='binary_crossentropy',metrics=['accuracy'])

     ##Training the network
     history=model.fit_generator(train_generator,steps_per_epoch=100,epochs=30,validation_data=validation_generator,validation_steps=50)
     model.save("cats_dogs_small.h5")

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

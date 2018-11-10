#!/usr/bin/env python
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator as IDG
import os


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import cm

if __name__=="__main__":
	train_cats_dir='/Users/han/keras_use/examples/CNN/dogsvscats/train/cats'
	fnames=[os.path.join(train_cats_dir,fname) for fname in os.listdir(train_cats_dir)]
#	print fnames
	img_path=fnames[3]
	print img_path
	img=image.load_img(img_path,target_size=(150,150))
#	print type(img)
	x=image.img_to_array(img)
	x=x.reshape((1,)+x.shape)
#	print type(x),x.shape

	##Image data augmentation
	i=0
	datagen=IDG(rotation_range=60,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')
	for batch in datagen.flow(x,batch_size=1):
	    plt.figure(i)
	    imgplot=plt.imshow(image.array_to_img(batch[0]))
	    plt.savefig(str(i)+"cat.png")
	    i+=1
	    if i % 4==0: break
	plt.show()


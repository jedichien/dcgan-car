from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
import os
import glob
import random
import math
from PIL import Image
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD

"""
load training data
"""
def list_image_pathes(trainDir, testDir):
    return glob.glob(trainDir+"/*.jpg"), glob.glob(testDir+"/*.jpg")

def random_images(pathes, target_size, batch_size=32):
    rRange = random.randint(0, len(pathes)-batch_size-1)
    randomPathes = pathes[rRange:rRange+batch_size]
    images = [image.img_to_array(img) for img in 
                  [image.load_img(path, target_size=target_size) for path in randomPathes]]
    return (np.array(images).astype(np.float32)-127.5)/127.5

def combine_generated_images(generated_images):
    num = generated_images.shape[0]
    gridWidth = int(math.sqrt(num))
    gridHeight = int(math.ceil(float(num)/gridWidth))
    shape = generated_images.shape[1:3]
    image = np.zeros((gridHeight*shape[0], gridWidth*shape[1], 3), dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/gridWidth)
        j = index % gridWidth
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1],:] = img[:, :, :]
    return image

"""
build model
"""
def generator_model(input_dim=128):
    model = Sequential()
    
    model.add(Dense(input_dim=input_dim, units=1024))
    model.add(Activation('tanh'))
    
    model.add(Dense(1024*4*4))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    
    model.add(Reshape((4, 4, 1024), input_shape=(1024*4*4,)))
    # 8x8
    model.add(UpSampling2D(size=(4, 4)))
    model.add(Conv2D(512, strides=(2, 2), kernel_size=5, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    # 16x16
    model.add(UpSampling2D(size=(4, 4)))
    model.add(Conv2D(256, strides=(2, 2), kernel_size=5, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    # 32x32
    model.add(UpSampling2D(size=(4, 4)))
    model.add(Conv2D(128, strides=(2, 2), kernel_size=5, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    
    # 64x64
    model.add(UpSampling2D(size=(4, 4)))
    model.add(Conv2D(3, strides=(2, 2), kernel_size=5, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    
    return model

def discriminator_model(input_shape=(64, 64, 3)):
    model = Sequential()
    
    model.add(
            Conv2D(64, kernel_size=2, strides=1,
            padding='same',
            input_shape=input_shape)
            )
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, kernel_size=2, strides=1))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(256, kernel_size=2, strides=1))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(512, kernel_size=2, strides=1))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model

def concatenate_g_d(g, d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    return model

"""
optimizer
"""
g_optim = SGD(lr=0.0002, momentum=0.5, nesterov=True)
d_optim = SGD(lr=0.0002, momentum=0.5, nesterov=True)

# realImgs = random_images(pathes, target_size)
"""
create model
"""
input_dim = 100
g_model = generator_model(input_dim=input_dim)
d_model = discriminator_model()
g_model.load_weights('generator_weights.h5')
d_model.load_weights('discriminator_weights.h5')
concatenate_model = concatenate_g_d(g_model, d_model)

def train():
	"""
	configuration
	"""
	g_model.compile(loss='binary_crossentropy', optimizer='SGD')
	concatenate_model.compile(loss='binary_crossentropy', optimizer=g_optim)
	d_model.trainable = True
	d_model.compile(loss='binary_crossentropy', optimizer=d_optim)
	
	"""
	train
	"""
	trainDir = os.path.join('data', 'cars_train')
	testDir = os.path.join('data', 'cars_test')
	target_size=(64, 64)
	batch_size = 128
	train_p, test_p = list_image_pathes(trainDir, testDir)
	pathes = train_p + test_p

	for epoch in range(10, 1000):
	    # traverse all of pathes
	    for times in range(len(pathes)/batch_size):
	        # mimic image
	        noises = np.random.uniform(-1, 1, size=(batch_size, input_dim))
	        generated_images = g_model.predict(noises, verbose=0)
	        
	        # store currency
	        if times % 10 == 0:
	            combinedImg = combine_generated_images(generated_images)
	            combinedImg = combinedImg*127.5+127.5
	            Image.fromarray(combinedImg.astype(np.uint8)).save("{}_epoch_{}_times.png".format(epoch, times))
	        
	        # train discriminator
	        realImgs = random_images(pathes, target_size=target_size, batch_size=batch_size)
	        X = np.concatenate((realImgs, generated_images))
	        y = [1]*batch_size + [0]*batch_size
	        d_loss = d_model.train_on_batch(X, y)
	        
	        # train generated model
	        noises = np.random.uniform(-1, 1, size=(batch_size, input_dim))
	        d_model.trainable = False
	        g_loss = concatenate_model.train_on_batch(noises, [1]*batch_size)
	        d_model.trainable = True
	        print "epoch={}, times={}, d_loss={}, g_loss={}".format(epoch, times, d_loss, g_loss)
	        
	        if times % 50 == 0:
	            g_model.save_weights('generator_weights.h5', True)
	            d_model.save_weights('discriminator_weights.h5', True)

if __name__ =='__main__':
    train()



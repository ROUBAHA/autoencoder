
'''
@author:OUBAHA Rachid & El Makhroubi Mohammed
@date:28/06/2020
@title : An autoencoder with CNN | Python

 


'''


#import lib

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import keras
from keras.layers import Input, add
from keras.datasets import mnist
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.models import Model

from tensorflow.keras.callbacks import TensorBoard
#TensorBoard
#tensorboard --logdir logs
#http://localhost:6006/
tensorBoard=TensorBoard(log_dir='logs')



#load DATA and normalize
#We don't need the labels as the autoencoders are unsupervised network
#We want the pixels values between 0 and 1 instead of between 0 and 255
(X_train, _), (X_test, _) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train.astype("float32")/255.
X_test = X_test.astype("float32")/255.

#Create noisy dat
noise_factor = 0.5
X_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape) 
X_test_noisy = X_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape) 

X_train_noisy = np.clip(X_train_noisy, 0., 1.)
X_test_noisy = np.clip(X_test_noisy, 0., 1.)

#Create the network

#The first network is the most simple autoencoder. It has three layers : Input - encoded - decoded
'''
pool_size: integer or tuple of 2 integers, window size over which to take the maximum.
 (2, 2) will take the max value over a 2x2 pooling window. If only one integer is specified,
  the same window length will be used for both dimensions.

 padding: One of "valid" or "same" (case-insensitive). "valid" adds no zero padding.
  "same" adds padding such that if the stride is 1, 
  the output shape is the same as input shape.
'''

x = Input(shape=(28, 28, 1))

#-1 Encoder
conv1_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
pool1 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv1_1)
conv1_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
h = MaxPooling2D(pool_size=(2, 2), padding='same')(conv1_2)


#-2 Decoder
conv2_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(h)
up1 = UpSampling2D(size=(2, 2))(conv2_1) # the simplest way to upsample an input is to double each row and column.
conv2_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1)
up2 = UpSampling2D(size=(2, 2))(conv2_2)
r = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2)

autoencoder = Model(inputs=x, outputs=r)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


#Train the network
epochs = 3
batch_size = 128

history = autoencoder.fit(X_train_noisy, X_train, batch_size=batch_size, 
	epochs=epochs, verbose=1, validation_data=(X_test_noisy, X_test),callbacks=[tensorBoard])

#save model
autoencoder.save('saved_model/my_model')
#predict
decoded_imgs = autoencoder.predict(X_test_noisy)
#plot
n = 10
plt.figure(figsize=(20, 6))
for i in range(n):
    # display original
    ax = plt.subplot(3, n, i+1)
    plt.imshow(X_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    
    # display reconstruction
    ax = plt.subplot(3, n, i+n+1)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
plt.show()


#Plot the losses
print(history.history.keys())

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()


#------------------------------|THE END|----------------------------------------
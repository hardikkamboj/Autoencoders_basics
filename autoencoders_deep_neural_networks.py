# -*- coding: utf-8 -*-

from keras.datasets import mnist
import numpy as np
from keras.models import Model
from keras.layers import Dense,Input
import matplotlib.pyplot as plt

(X_train,_),(X_test,_) = mnist.load_data()


#normalizing the images
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
X_train /= 255
X_test /= 255

#converting it to a col matrix
X_train = np.reshape(X_train, newshape = (len(X_train),784))
X_test = np.reshape(X_test,newshape = (len(X_test),784))

print('The shape of X_train is ',X_train.shape)
print('The shape of X_test is ',X_test.shape)


#dimension for encoded version of image
encoded_dim = 32


#autoencoder_model
inputs = Input(shape = (784,))

#encoded layer
encoded = Dense(units = 128,activation = 'relu')(inputs)
encoded = Dense(units = 64,activation = 'relu')(encoded)
encoded = Dense(units = 32,activation = 'relu')(encoded)

#decoded_layer
decoded = Dense(units = 64,activation = 'relu')(encoded)
decoded = Dense(units = 128,activation = 'relu')(decoded)
decoded = Dense(units = 784,activation = 'sigmoid')(decoded)

#compiling the above layers in a model
autoencoder = Model(inputs = inputs,outputs = decoded)


#separate model for encoder
encoder = Model(inputs = inputs,outputs = encoded)


#separate model for decoder
encoded_layer = Input(shape = (encoded_dim,)) 
decoded_layer = autoencoder.layers[-1](encoded_layer)

decoder = Model(inputs = encoded_layer,outputs = decoded_layer)

#compiling the autoencoder
autoencoder.compile(loss = 'binary_crossentropy',optimizer = 'adadelta')


hist = autoencoder.fit(X_train,X_train,epochs = 100,batch_size = 256,
                       shuffle = True,validation_data = (X_test,X_test))


#accuracy_graph
plt.figure(figsize = (20,8))
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.xlabel('Epochs')
plt.ylabel("Loss")
plt.legend(['train_loss','val_loss'])
plt.show()

#getting the encoded images, and decoded images
encoded_images = encoder.predict(X_test)
decoded_images = decoder.predict(encoded_images)


def show_decoded_images(index,decoded_images):
    n = 10#number of images
    plt.figure(figsize = (20,4))
    for i in range(n):
        plt.subplot(2,n,i+1)
        plt.imshow(np.reshape(X_test[index + i],(28,28)))
        plt.axis('off')
        
        plt.subplot(2,n,i+n+1)
        plt.imshow(np.reshape(decoded_images[index + i],(28,28)))
        plt.axis('off')
    plt.show()

show_decoded_images(10,decoded_images)    










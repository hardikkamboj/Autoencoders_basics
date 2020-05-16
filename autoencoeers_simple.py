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
encoded = Dense(units = encoded_dim,activation = 'relu')(inputs)

#decoded_layer
decoded = Dense(units = 784,activation = 'sigmoid')(encoded)

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


hist = autoencoder.fit(X_train,X_train,epochs = 50,batch_size = 256,
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


#the best loss we have is 0.10
# to get better loss we will use more epochs
# to tackle overfitting problem for more number of epochs, we will use regularization

from keras import regularizers


#dimension for encoded version of image
encoded_dim = 32


#autoencoder_model
inputs2 = Input(shape = (784,))

#encoded layer
encoded2 = Dense(units = encoded_dim,activation = 'relu'
                ,activity_regularizer = regularizers.l1(10e-5))(inputs2)

#decoded_layer
decoded2 = Dense(units = 784,activation = 'sigmoid')(encoded2)

#compiling the above layers in a model
autoencoder2 = Model(inputs = inputs2,outputs = decoded2)


#separate model for encoder
encoder2 = Model(inputs = inputs2,outputs = encoded2)


#separate model for decoder
encoded_layer2 = Input(shape = (encoded_dim,)) 
decoded_layer2 = autoencoder2.layers[-1](encoded_layer2)

decoder = Model(inputs = encoded_layer2,outputs = decoded_layer2)

#compiling the autoencoder
autoencoder2.compile(loss = 'binary_crossentropy',optimizer = 'adadelta')

#now we are going to train for 100 epochs
hist = autoencoder2.fit(X_train,X_train,epochs = 100,batch_size = 256,
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
encoded_images2 = encoder.predict(X_test)
decoded_images2 = decoder.predict(encoded_images)

show_decoded_images(10,decoded_images2)

plt.imshow(np.resize(decoded_images[0],(28,28)))








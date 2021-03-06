from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

(X_train,_),(X_test,_) = mnist.load_data()

X_train = X_train.astype(np.float32) / 255
X_test = X_test.astype(np.float32) / 255

X_train = np.reshape(X_train,(len(X_train),28,28,1))
X_test = np.reshape(X_train,(len(X_test),28,28,1))

X_train_nosiy = X_train + noise_factor*np.random.normal(size = X_train.shape)
X_test_noisy = X_test + noise_factor*np.random.normal(size = X_test.shape)


def visualize_noisy_dataset():
    pass

visualize_noisy_dataset(index = 10)



input_image = Input(shape = (28,28,1))

x = Conv2D(16,(3,3),activation = 'relu',padding = 'same')(input_image) #(28,28,16)
x = MaxPooling2D((2,2))(x) #(14,14,16)
x = Conv2D(32,(3,3),activation = 'relu',padding = 'same')(x) #(14,1432)
x = MaxPooling2D((2,2))(x) #(7,7,8)
encoded = Conv2D(64,(3,3),activation = 'relu',padding = 'same')(x) #(7,7,64)

x = Conv2D(64,(3,3),activation = 'relu',padding = 'same')(encoded) #(7,7,64)
x = UpSampling2D((2,2))(x) #(14,14,64)
x = Conv2D(32,(3,3),activation = 'relu',padding = 'same')(x) #(14,14,32)
x = UpSampling2D((2,2))(x) #(28,28,32)
decoded = Conv2D(1,(3,3),activation = 'relu',padding = 'same')(x) #(28,28,1)

autoencoder = Model(inputs = input_image,outputs = decoded)


autoencoder.compile(loss = 'binary_crossentropy',optimizer = 'adadelta')

hist = autoencoder.fit(X_train_noisy,X_train,epochs = 50,batch_size = 128,
                       validation_data = (X_test,X_test),verbose = 1)


#mkaing a model for encoder
encoder = Model(inputs = input_image,outputs = encoded)

 
#loss graph
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss graph')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train_loss','Valid_loss'])
plt.show()


decoded_images = autoencoder.predict(X_test)
encoded_images = encoder.predict(X_test)

def show_decoded_images(index,decoded_images):
    n = 10#number of images
    plt.figure(figsize = (20,4))
    for i in range(n):
        plt.subplot(2,n,i+1)
        plt.imshow(np.reshape(X_test_nosiy[index + i],(28,28)))
        plt.gray()
        plt.axis('off')
        
        plt.subplot(2,n,i+n+1)
        plt.imshow(np.reshape(decoded_images[index + i],(28,28)))
        plt.axis('off')
    plt.show()

show_decoded_images(10,decoded_images) 











from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import  to_categorical
(train_data, train_labels) ,(test_data, test_labels) = mnist.load_data()
import matplotlib.pyplot as plt

train_images = train_data.reshape((60000, 28*28))
train_images= train_images.astype('float32')/255

test_images = test_data.reshape((10000, 28*28))
test_images = test_images.astype('float32')/255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28, )))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop', loss= 'categorical_crossentropy', metrics= ['accuracy'] )

network.fit(train_images, train_labels, epochs= 5 ,batch_size=128)
print(train_data)
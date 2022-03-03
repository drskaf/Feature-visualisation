import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os
import numpy as np
import matplotlib.pyplot as plt

from keras.applications.vgg16 import VGG16
from keras import models

model = VGG16(include_top=True, weights='imagenet')

img_path = '4C.png'
img = load_img(img_path, target_size=(224, 224))
x = img_to_array(img)
x = x.reshape((1,) + x.shape)
x /= 255.0

layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(x)

# Getting Activations of first layer
first_layer_activation = activations[0]

# shape of first layer activation
print(first_layer_activation.shape)

# 6th channel of the image after first layer of convolution is applied
plt.matshow(first_layer_activation[0, :, :, 1], cmap='gray')
plt.show()


second_layer_activation = activations[1]
plt.matshow(second_layer_activation[0, :, :, 1], cmap='gray')
plt.show()

third_layer_activation = activations[2]
plt.matshow(third_layer_activation[0, :, :, 1], cmap='gray')
plt.show()

fourth_layer_activation = activations[3]
plt.matshow(fourth_layer_activation[0, :, :, 1], cmap='gray')
plt.show()

fifth_layer_activation = activations[4]
plt.matshow(fifth_layer_activation[0, :, :, 1], cmap='gray')
plt.show()

sixth_layer_activation = activations[5]
plt.matshow(sixth_layer_activation[0, :, :, 1], cmap='gray')
plt.show()

seventh_layer_activation = activations[6]
plt.matshow(seventh_layer_activation[0, :, :, 1], cmap='gray')
plt.show()

eighth_layer_activation = activations[7]
plt.matshow(eighth_layer_activation[0, :, :, 1], cmap='cividis')
plt.show()

img_path = '4C.png'
img = load_img(img_path, target_size=(224, 224))
x = img_to_array(img)
x = x.reshape((1,) + x.shape)
x /= 255.0

img_path = 'CXR.png'
img = load_img(img_path, target_size=(224, 224))
x = img_to_array(img)
x = x.reshape((1,) + x.shape)
x /= 255.0

layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(x)

eighth_layer_activation = activations[5]
plt.matshow(eighth_layer_activation[0, :, :, 1], cmap='cividis')
plt.show()

img_path = 'perfusion.png'
img = load_img(img_path, target_size=(224, 224))
x = img_to_array(img)
x = x.reshape((1,) + x.shape)
x /= 255.0

layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(x)

first_layer_activation = activations[0]

# shape of first layer activation
print(first_layer_activation.shape)

# 6th channel of the image after first layer of convolution is applied
plt.matshow(first_layer_activation[0, :, :, 1])
plt.show()


second_layer_activation = activations[1]
plt.matshow(second_layer_activation[0, :, :, 1])
plt.show()

third_layer_activation = activations[2]
plt.matshow(third_layer_activation[0, :, :, 1])
plt.show()

fourth_layer_activation = activations[3]
plt.matshow(fourth_layer_activation[0, :, :, 1])
plt.show()

fifth_layer_activation = activations[4]
plt.matshow(fifth_layer_activation[0, :, :, 1])
plt.show()

sixth_layer_activation = activations[5]
plt.matshow(sixth_layer_activation[0, :, :, 1], cmap='gray')
plt.show()

seventh_layer_activation = activations[6]
plt.matshow(seventh_layer_activation[0, :, :, 1], cmap='gray')
plt.show()

eighth_layer_activation = activations[7]
plt.matshow(eighth_layer_activation[0, :, :, 1], cmap='cividis')
plt.show()

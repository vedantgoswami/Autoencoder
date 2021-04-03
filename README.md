# Autoencoder
In this repository i will show how the Autoencoder works and it's application of denoising the images.

 The idea of auto encoders is to allow a feed forward neural network to figure out how to best encode and decode certain data.Autoencoders are an unsupervised learning approach to some of issues and techniques such as dimensionality reduction, noise reduction and data compression.
 <p>
 But we will focus on basic bulding of autoencoder which comprises of an encoder and a decoder. And then wee will use this autoencoder for image denoising.
 I will be using the MNIST image dataset to keep it simple.
 </p>
 
 ### Structure of AutoEncoder
 <p align="center">
 <img align="center" src="https://github.com/vedantgoswami/Autoencoder/blob/main/Images/model.png">
 </p>

In the above diagram the encoder part compress the image passed in input by using the convolutional layers and pooling layers.The final output of the encoder will a compressed representation of image which is then passed to the decoder as input. The decoder will then use the Upsampling layers or Transposed convolutional layers to reconstruct the original image back.

<b> What is UpSampling? </b><br>
As Max pooling is a sampling strategy that picks the maximum value from a window. Upsampling is just reverse of it each value can be surrounded with zeros to upsample the layer.<br>
<p align="center">
<img  src="https://github.com/vedantgoswami/Autoencoder/blob/main/Images/upsampling.png" width=636dp height=312dp>
 </p>
 <b> What is Transposed Convolution? </b><br>
 <b>Convolution + Upsampling</b><br>
 Transposed Convolutions are used to upsample the input feature map to a desired output feature map using some learnable parameters.
 <p align="center">
 <img src="https://github.com/vedantgoswami/Autoencoder/blob/main/Images/Transposed%20convolution.png"> 
 </p>

### Let's Build The Model
We Require the following models:
```
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import cv2
import random
import numpy as np
```
<br>

We are using MNIST images dataset, it comprises of 70000 28 pixels by 28 pixels images of handwritten digits and 70000 vectors containing information on which digit each one is.

<h3> Loading Dataset</h3>

```
(x_train,y_train),(x_test,y_test) = keras.datasets.mnist.load_data()
```

<h3> Normalising Data </h3>
Normalising data by dividing it by 255 should improve activation functions performance - sigmoid function works more efficiently with data range 0.0-1.0.
```
x_train = x_train/255.0
x_test = x_test/255.0
```

<h3> Encoder Part </h3>

```
    encoder_input = keras.Input(shape=(28, 28, 1), name='img')
    x = keras.layers.Flatten()(encoder_input)
    encoder_output = keras.layers.Dense(level*level, activation="relu")(x)
    encoder = keras.Model(encoder_input, encoder_output, name='encoder')
```

The Encoder model takes images as input of shape <b>(28 x 28)</b> and then flattening it to vector of pixels of shape <b>(786,)</b>.
### My Encoder Sample
<p>
<img src="https://github.com/vedantgoswami/Autoencoder/blob/main/Images/My%20model.png">
 </p>
As we can see that there is a data loss due to compression as the image is diminised but still it is recognizable as 7.<br>
The above image is 28

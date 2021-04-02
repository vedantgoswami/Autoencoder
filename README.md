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

### My Encoder Sample
<p>
<img src="https://github.com/vedantgoswami/Autoencoder/blob/main/Images/My%20model.png">
 </p>
As we can see that there is a data loss due to compression as the image is diminised but still it is recognizable as 7.<br>
The above image is 28

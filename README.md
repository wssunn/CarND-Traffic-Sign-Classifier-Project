**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./img/visualization.jpg "Visualization"
[image2]: ./img/grayscale.jpg "Grayscaling"
[image3]: ./img/traffic_sign.jpg
[image4]: ./img/resnet.jpg


### Data Set Summary & Exploration

#### 1. A basic summary of the data set

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Visualization of the dataset

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Preprocessing of image data

As a first step, I decided to convert the images to grayscale because this reduces the color channel (hence reduce the size of data and speeds up training)

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because this helps with the training


#### 2. Model architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							|
| Preprocessing			| 32x32x1 Grey image							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 				|
| ResBlock 			    | output 14x14x32      							|
| ResBlock 			    | output 14x14x32      							|
| ResBlock 			    | output 14x14x32      							|
| Max pooling	      	| 2x2 stride,  outputs 7x7x32 					|
| Fully connected		| input 1568; output 64        					|
| RELU					|												|
| Softmax				| etc.        									|
|						|												|
|						|												|
 
It is based on the resnet, here is what the building block looks like:

![alt text][image4]

#### 3. Training of the model

To train the model, I used an Adam Optimizer, with 35 Epochs and batch size 256, learning rate 0.01

The first architecture chosen is LeNet with 64 and 128 CNN depth. The model gives a 0.951 validation accuracy and 0.933 test accuracy (which is the same as the ResNet). However, it takes a very long time to train this model, it has a lot of parameters which could cause it to overfit the model.

I have tried Google's InceptionNet, ResNet as well as InceptionRes Net. By experiment, ResNet offers a good performance with very quick training (with least number of parameters.) My final model results were:
* training set accuracy of 0.994
* validation set accuracy of 0.951
* test set accuracy of 0.933

A CNN works well with image recognition because it incorporates weight sharing and statistical invariance. This greatly reduces number of parameters of the network and imporves accuracy. Dropout helps reducing overfitting, gradient exploding and gradient vanishing.

Typical adjustments could include 
- choosing a different model architecture, 
- adding or taking away layers (pooling, dropout, convolution, etc), 
- using an activation function or changing the activation function. 

One common justification for adjusting an architecture would be due to overfitting or underfitting. 
- A high accuracy on the training set but low accuracy on the validation set indicates over fitting; 
- a low accuracy on both sets indicates under fitting.


### Test a Model on New Images

#### 1. Test on German traffic signs found on the web 

Here are five German traffic signs that I found on the web:

![alt text][image3] 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 20 km/h	      		| 20 km/h						 				|
| 30 km/h	      		| 30 km/h						 				|
| 40 km/h	      		| 40 km/h						 				|
| 70 km/h	      		| 70 km/h						 				|
| 80 km/h	      		| 80 km/h						 				|
| 80 km/h cancelled     | 80 km/h cancelled								|


The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 100%.

The 50km/h sign could be a bit difficult to classify becuase the brightness of it is very low.

#### 3. Prediction of the model and its performance

The code for making predictions on my final model is located in the 28th cell of the Ipython notebook.

For all images, the model is completely sure about its prediction, with softmax possibility of 1.0



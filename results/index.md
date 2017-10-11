# Your Name <span style="color:red">(id)</span>

#Project 5: Deep Classification

## Overview
The project is related to 
> Using VGG-16 network to implement this project. The classification task we want to do is object categories.


## Implementation
1. load data
	* I writed in data_loader.py
	* The data size is large, so I resize the data and named dataset. Because this homework just want to classify the object captured by HandCam, 
 some images we won't use. So we can take them off.
2. train with VGG-16 network
	* I writed in train_vgg_16.py
	* I download the pre-trained model of VGG-16 (vgg_16.ckpt) to use. I remove the last fully connected layer and replace it with our own.
The output size is 24 (object categories).
	* Then we want to fine-tune the entire model.


## Installation
* tensorflow
* tensorflow.contrib.slim
* numpy
* tqdm
* >> python train_vgg_16.py

### Results

<table border=1>
<tr>
<td>
After training and testing, the testing accuracy is 0.6090325610519725.
</td>
</tr>

<tr>
<td>
<img src="results/results.png" alt="results" style="float:middle;">
</td>
</tr>

</table>



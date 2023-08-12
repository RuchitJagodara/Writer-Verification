# Writer-Verification
# Introduction
The Writer Verification challenge is a competition that involved identifying whether a given pair of handwritten text samples was written by the same person or two different persons with the help of Siamese neural network and OpenCV. In the accompanying Github repository, you will find method that we came up with to do so but this can be done much better by making a new neural network in which layers are set on the basis of alignments and darkness of a letter but because of lack of time during the event, it was very hard to do so but will make it perfect in future. 
# Dataset
The training set consists of 1352 folders, each containing a set of images written by the same person.
The validation set contains a set of images from 92 different writers, along with a file called 'val.csv' that contains pairs of image names and corresponding labels. A label of 1 indicates that the images were written by the same writer, and a label of 0 indicates that the images were written by different writers.
Test set contains images from 360 writers. In test.csv you are given name of image pairs. For output, you need to predict the label for given pair of images and submit the csv file in the format by editing the test.csv file.
# References
For our model, followong references were used.

1. [SigNet: Convolutional Siamese Network for Writer Independent Offline Signature Verification](https://arxiv.org/pdf/1707.02131v2.pdf)
2. [Attention based Writer Independent Verification](https://arxiv.org/pdf/2009.04532v3.pdf)

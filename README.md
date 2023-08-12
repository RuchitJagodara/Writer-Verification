# Writer-Verification
# Introduction
The Writer Verification challenge is a competition that involves identifying whether a given pair of handwritten text samples was written by the same person or two different persons. In the accompanying Github repository, you will find method that we came up with to do so but this can be done much better by making a new neural network in which layers are set on the basis of alignments and darkness of a letter but because of lack of time during the event, it was very hard to do so but will make it perfect in future. 
# Dataset
The training set consists of 1352 folders, each containing a set of images written by the same person.
The validation set contains a set of images from 92 different writers, along with a file called 'val.csv' that contains pairs of image names and corresponding labels. A label of 1 indicates that the images were written by the same writer, and a label of 0 indicates that the images were written by different writers.
Test set contains images from 360 writers. In test.csv you are given name of image pairs. For output, you need to predict the label for given pair of images and submit the csv file in the format by editing the test.csv file.
# References
For our baseline model, followong references were used and you are advised too look at the same.

SigNet: Convolutional Siamese Network for Writer Independent Offline Signature Verification
Attention based Writer Independent Verification

# Face Detection using Viola Jones

Face Detection in Python using the Viola-Jones algorithm on the CBCL Face Database published by MIT's Center for Biological and Computational Learning. 
Link to [paper](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf)

# Data

Each image is 19x19 and greyscale. There are 2,429 faces, 4,548 non-faces in the Training set and 472 faces, 23,573 non-faces in the Test set.

# Results

For hyper parameter value T=75 which defines the number of weak classsifier used, the model achieved an accuracy of 93%.

```
Testing Accuracy: 0.928709055876686
Testing Precision: 1.0
Testing Recall: 0.8574181117533719
Confusion matrix: tp : 445, tn : 519, fp : 0, fn : 74
```

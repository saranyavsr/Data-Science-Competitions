# Data-Science-Class-Competition

## Medical-Records-Classification

Medical abstracts describe the current conditions of a patient. Doctors routinely scan dozens or hundreds of abstracts each day as they do their rounds in a hospital and must quickly pick up on the salient information pointing to the patient’s malady. You are trying to design assistive technology that can identify, with high precision, the class of problems described in the abstract. In the given dataset, abstracts from 5 different conditions have been included: digestive system diseases, cardiovascular diseases, neoplasms, nervous system diseases, and general pathological conditions.

The goal of this competition is to develop predictive models that can determine, given a particular medical abstract, which one of 5 classes it belongs to.

### Dataset
* train.dat: 14442 records. Training set (class label, followed by a tab separating character and the text of the medical abstract).
* test.dat: 14438 records. Testing set (text of medical abstracts in lines, no class label provided).

## Movie-Ratings-Prediction

The goal of this competition to predict the ratings for 18,122 <user, movie> pairs provided in the test file (test.dat). Ratings are in the range of 1 to 5. You will achieve this by using the 54,543 ratings of 17615 distinct users provided as training data (train.dat).

### Dataset
Datasets used in this competition are a subset of the ciaoDVD dataset, containing ratings on DVDs from the dvd.ciao.co.uk website.

* train.dat: userId, movieId, movieRating (54543 instances)
* test.dat: userId, movieId (18122 instances)

## Image-Classification

Traffic congestion seems to be at an all-time high. Machine Learning methods must be developed to help solve traffic problems. The goal of this competition is to analyze features extracted from traffic images depicting different objects to determine their type as one of 11 classes, noted by integers 1-11: car, suv, small_truck, medium_truck, large_truck, pedestrian, bus, van, people, bicycle, and motorcycle. The object classes are heavily imbalanced. For example, the training data contains 10375 cars but only 3 bicycles and 0 people. Classes in the test data are similarly distributed. The input to analysis will not be the images themselves, but rather features extracted from the images.

### Dataset
An image can be can be described by many different types of features. In the training and test datasets, images are described as 887-dimensional vectors, composed by concatenating the following features:
- 512 Histogram of Oriented Gradients (HOG) features
- 256 Normalized Color Histogram (Hist) features
- 64 Local Binary Pattern (LBP) features
- 48 Color gradient (RGB) features
- 7 Depth of Field (DF) features

* train.dat: Training set (dense matrix, samples/images in lines, features in columns). 21186 records
* train.labels: Training class labels (integers, one per line).
* test.dat: Test set (dense matrix, samples/images in lines, features in columns). 5296 records


## Text-Clustering 

Implement the DBSCAN clustering algorithm. All objects in the training data set must be assigned to a cluster. Thus, you can either assign all noise points to cluster K+1 or apply post-processing after DBSCAN and assign noise points to the closest cluster.

### Dataset
Input data (provided as training data) consists of 8580 text records in sparse format.



# Data-Science-Class-Competition

## Medical-Text-Classification

The goal of this competition is to develop predictive models that can determine, given a particular medical abstract, which one of 5 classes it belongs to.

### Dataset
* train.dat: 14442 records Training set (class label, followed by a tab separating character and the text of the medical abstract).
* test.dat: 14438 records Testing set (text of medical abstracts in lines, no class label provided).
* format.dat: A sample submission with 14438 entries randomly chosen to be 1 to 5.


## Movie-Ratings-Prediction

The goal of this competition to predict the ratings for 18,122 <user, movie> pairs provided in the test file (test.dat). Ratings are in the range of 1 to 5. You will achieve this by using the 54,543 ratings of 17615 distinct users provided as training data (train.dat).

### Dataset
Datasets used in this competition are a subset of the ciaoDVD dataset, containing ratings on DVDs from the dvd.ciao.co.uk website.

* train.dat: userId, movieId, movieRating (54543 instances)
* test.dat: userId, movieId (18122 instances)

## Image-Classification (On-going Competition)

The goal of this competition is to analyze features extracted from traffic images depicting different objects to determine their type as one of 11 classes, noted by integers 1-11: car, suv, small_truck, medium_truck, large_truck, pedestrian, bus, van, people, bicycle, and motorcycle. The object classes are heavily imbalanced.

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


## Yahoo-Music-Recommendations (On-going Project)

Yahoo! Music offers a wealth of information and services related to many aspects of music. This dataset represents a snapshot of the Yahoo! Music community's preferences for various musical items. A distinctive feature of this dataset is that user ratings are given to entities of four different types: tracks, albums, artists, and genres. In addition, the items are tied together within a hierarchy. That is, for a track we know the identity of its album, performing artist and associated genres. Similarly we have artist and genre annotation for the albums. We provide four different versions of the dataset, differing by their number of ratings, so that researchers can select the version best catering for their needs and hardware availability. In addition, one of the datasets is oriented towards a unique learning-to-rank goal, unlike the predictive / error-minimization criteria common at recommender systems. The dataset contains ratings provided by true Y! Music customers during 1999-2009. Both users and items (tracks, albums, artists and genres) are represented as meaningless anonymous numbers so that no identifying information is revealed. ) The important and unique features of the dataset are fourfold: It is of very large scale compared to other datasets in the field, e.g, 13X larger than the Netflix prize dataset. It has a very large set of items – much larger than any similar dataset, where usually only number of users is large. There are four different categories of items, which are all linked together within a defined hierarchy. It allows performing session analysis of user activities. We expect that the novel features of the dataset will make it a subject of active research and a standard in the field of recommender systems. In particular, the dataset is expected to ignite research into algorithms that utilize hierarchical structure annotating the item set.

Dataset Link - https://webscope.sandbox.yahoo.com/catalog.php?datatype=c

### Challenges
* Scale: biggest public dataset ever. 1 million user, 0.6 million items, 300 million ratings
* Hierarchical item relation: song belong to albums, albums belong to artists. All of them are annotated with genre tags.
* Rich meta data: over 900 genres
* Fine temporal resolution: no previous challenge provided time in addition to date.


#### Track 1
Goal is to predict ratings that users give to songs
* Blending of multiple techniques
* Matrix factorization models
* Nearest neighbor models
* Restricted Bolzmann machines
* Temporal modelings

#### Track 2
Goal is to predict whether the user will rate a song or not.
* Importance sampling of negative instances
* Taxonomical modelings
* Use of pairwise ranking objective functions


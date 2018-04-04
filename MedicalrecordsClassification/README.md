# Medical Records Classification

The purpose of the competition is to predict the medical class using machine- learning algorithms. The model uses tf-idf vectorizer as a baseline, Latent Semantic Analysis (LSA) and SGDClassifier written in Python using the scikit- learn library. 

###  Step 1 : Analyzing Document
First approach was to split the medical record document (train.dat) into records and their respective classes and the same method was applied to testing document (test.dat) to retrieve just the records.

###  Step 2 : Text Preprocessing

The training data is in plain-text format that needs to be preprocessed. The bag-of-words approach is used here, where each unique word in a text will be represented by one number. build_analyzer() function will split a message into its individual words, and return a list. This function uses stemming, lemmatizing and postag to remove stop-words, and returns a list of the remaining words, or tokens (using NLTK library)

###  Step 3 : Vectorization using Pipeline

The classification algorithm will need some sort of feature vector in order to perform the classification task. CountVectorizer is used to convert the text collection into a matrix of token counts and returns a Document-Term matrix of [n_samples, n_features]. TfidfTransformer technique is used for document similarity. Pipeline - Instead of manually running through each of the Vectorization steps, and pipeline is used to extracts the text documents, tokenizes them, counts the tokens, and then performs a tf–idf transformation before passing the resulting features along with different classifiers.

###  Step 4 : Split Train/Test Data & Run Classifier Algorithms

Training data is split into train (70%) and test data (30%) to test the accuracy of different Classifiers. Naive Bayes, SGDClassifier, Logistic Regression, KNN, Random Forest with default parameters was trained and tested to get the accuracy of the classifiers. SGDClassifier achieved the highest accuracy of 58%, followed by Logistic Regression (57%) and Naive Bayes (55%). SGDClassifier is chosen for further improvement for the above reason.

###  Step 5 : Tuning Parameters

Support Vector Machine (SVM) is one of the state-of-art methods is text classification. Stochastic gradient descent-based implementation in sklearn is used to develop the model. Also a sentence has composed a series of words, the relationship between word is also significant. To capture the relationship between words, 4-gram is usedduring convert document to feature representation. To improve the accuracy further, Hyper parameter tuning was done on SGDClassifier. The output of the model is dependent on an interaction between alpha and the number of epochs (n_iter). Using parameters of ’l2’ penalty, alpha value 0.0006 and n_iter to 10 & set random_state to 42, achieving the accuracy of 61%.

###  Step 6 : Dimensionality Reduction using Truncated SVD

Final approach was to use just the Text mining algorithm LSA and TfidfVectorizer. The main goal of TruncatedSVD is to reduce dimensionality and maintain the varience between samples. TfidfVectorizer is used as it combines all the options of CountVectorizer and TfidfTransformer in a single model. It strips out “stop words”, filter out terms that occur in more than half of the docs (max_df=0.5), filters= out terms that occur in only one document (min_df=2), select the 10,000 most frequently occuring words in the corpus, normalizes the vector (L2 norm of 1.0) to normalize the effect of document length on the tf-idf values. Then project the tfidf vectors onto the first N principal components (1000). Though this has significantly fewer features than the original tfidf vector, they are stronger features, and the accuracy is higher. Explained variance of SVD was 56% and improving by the accuracy to 64%.

###  Step 7 : Result

Re-trained model on full training set, using best parameters and applied re-trained model to test set. The output predicted is then stored onto output.dat, achieving .783 F1 score on Leaderboard. why we use F1 score? 99% accuracy does not imply 99% correct predictions unless the relative frequencies of the test cases are the same as in the real application situation. So to deal with the imbalanced data, F1 score is used as a measure to check the accuracy.

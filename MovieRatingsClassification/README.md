# Movie Ratings Prediction

The purpose of this competition is to predict the movie rating for the given userId and movieId and improve RMSE than what the “baseline” (i.e. plain user-based collaborative filtering with cosine similarity) can do. The method uses ML- Ensemble, a Python library for memory-efficient parallel ensemble learning with a Scikit-learn API. ML-Ensemble deploys a neural network-like API for building ensembles of several layers, and can accommodate a great variety of ensemble architectures. The notebook uses four base regressors - XGRegressor, RandomForest Regressor, Lasso, ElasticNet. Then XGBoostRegressor and RandomForest Regressor are used as meta learner and combine the predictions from the base models.


###  Step 1 : Analyzing Document

First approach was to split the training dataset (train.dat) into X as userId, movieId and Y as rating (target) and the same method was applied to testing data.

###  Step 2 : Preprocessing and splitting the train/test data

MinMaxScaler is used to normalize the data. userId and movieId are scaled to a fixed range between 0 to 1 so that it is easy to compare similarities between features based on certain distance measures. It also suppresses the effect of outliers. Training data is split using train_test_split to train (70%) and test data (30%) to develop a recommendation model.

###  Step 3 : Base learners performance

Lasso, ElasticNet, RandomForest Regressor, XGRegressor are set as Base learners (estimators) to test the data and their rmse values - ls : 0.9124, el : 0.9122, rf : 0.9101, gb : 0.8883 are predicted accordingly. All of their rmse’s are relatively close. However, they seem to capture different aspects of the feature space, as shown by the low correlation of their predictions.

###  Step 4 : Comparing base learners

To facilitate base learner comparison, ML-Ensemble implements a randomized grid search class that allows specification of several estimators (and preprocessing pipelines) in one grid search. Tuned models are compared using grid search and their optimal parameters are obtained.

###  Step 5 : Comparing meta learners

Upon comparing, GBM and Randomforest are chosen as the meta learner and are cloned internally so to get the fitted ones. The ensemble will implement is the Super Learner, also known as a stacking ensemble. After instantiation, ensemble can be used like any other Scikit-learn estimator. Predictions are generated, achieving .887 rmse score.

###  Step 6 : Result

Re-trained model on full training set, using ensemble and applied re-trained model to test set. The movie ratings predicted are then stored onto predictions.dat, achieving rmse (50%) 1.0544 on Leaderboard.

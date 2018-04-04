
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import time
get_ipython().magic('pylab inline')
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDClassifier

from sklearn.pipeline import make_pipeline

from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report

pd.set_option('max_colwidth', 1000)

trainrecord = pd.read_csv('/Users/whiplash/SJSU/Semester 2/CMPE 255/Assignments/Program 2/data/train.txt', 
                          header=None, delimiter=' ')

trainlabels = pd.read_csv('/Users/whiplash/SJSU/Semester 2/CMPE 255/Assignments/Program 2/data/train.labels', 
                          header=None, names =['labels'])

testrecord = pd.read_csv('/Users/whiplash/SJSU/Semester 2/CMPE 255/Assignments/Program 2/data/test.txt', 
                          header=None, delimiter=' ')


# In[2]:


print(trainlabels.groupby("labels").size())

sns.countplot(x="labels", data=trainlabels)


# In[3]:


from xgboost import XGBClassifier

target = []
for x, value in np.ndenumerate(trainlabels):
    target.append(value)

X = trainrecord
y = target


# In[4]:


# Split into train/test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[6]:


from scipy.stats import boxcox
from scipy.stats import skew
from scipy.stats import randint
from scipy.stats import uniform

from xgboost import XGBRegressor

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso, ElasticNet, Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import make_scorer 
from sklearn.base import BaseEstimator, RegressorMixin

# neural networks
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers

# ignore Deprecation Warning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


# In[8]:


import sklearn.pipeline

select = sklearn.feature_selection.SelectKBest(k=100)
clf = sklearn.ensemble.RandomForestClassifier()

steps = [('feature_selection', select),
        ('random_forest', clf)]

pipeline = sklearn.pipeline.Pipeline(steps)


# In[11]:


# fit your pipeline on X_train and y_train
pipeline.fit( X_train, y_train)

# call pipeline.predict() on your X_test data to make a set of test predictions
y_prediction = pipeline.predict(X_test)

# test your predictions using sklearn.classification_report()
report = sklearn.metrics.classification_report(y_test, y_prediction)

# and print the report
print(report)


# In[10]:


import sklearn.grid_search


parameters = dict(feature_selection__k=[100, 200], 
              random_forest__n_estimators=[50, 100, 200],
              random_forest__min_samples_split=[2, 3, 4, 5, 10])

cv = sklearn.grid_search.GridSearchCV(pipeline, param_grid=parameters)

cv.fit(X_train, y_train)
y_predictions = cv.predict(X_test)
report = sklearn.metrics.classification_report( y_test, y_predictions )


# In[12]:


pipeline.fit(X, y)

# call pipeline.predict() on your X_test data to make a set of test predictions
y_prediction = pipeline.predict(testrecord)


# In[13]:


np.savetxt('prediction.dat', y_prediction, delimiter=" ", fmt="%s")


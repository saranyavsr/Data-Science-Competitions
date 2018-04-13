
# coding: utf-8

# In[75]:


import pandas as pd
import numpy as np
import operator

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler


from collections import Counter
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import cross_val_score
import warnings
import time

import matplotlib.pyplot as plt
from sklearn import svm

from sklearn.feature_selection import VarianceThreshold

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier

from mlxtend.classifier import StackingClassifier

from sklearn.multiclass import OneVsRestClassifier
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek


# In[2]:


pd.set_option('max_colwidth', 1000)

trainrecord = pd.read_csv('/Users/whiplash/SJSU/Semester 2/CMPE 255/Assignments/Program 2/data/train.txt', 
                          header=None, delimiter=' ')

trainlabels = pd.read_csv('/Users/whiplash/SJSU/Semester 2/CMPE 255/Assignments/Program 2/data/train.labels', 
                          header=None, names =['labels'])

testrecord = pd.read_csv('/Users/whiplash/SJSU/Semester 2/CMPE 255/Assignments/Program 2/data/test.txt', 
                          header=None, delimiter=' ')


# In[77]:


target = []
for x, value in np.ndenumerate(trainlabels):
    target.append(value)

X = trainrecord
y = target
Xtestfinal = testrecord


# In[78]:


print(type(X))


# In[79]:


dataset = []
labels = []
for row, x in enumerate(y):
    if y[row] == 1:
        dataset.append(X.iloc[row])
        labels.append(y[row])
    if y[row] == 2:
        dataset.append(X.iloc[row])
        labels.append(y[row])
    if y[row] == 3:
        dataset.append(X.iloc[row])
        labels.append(y[row])
    if y[row] == 4:
        dataset.append(X.iloc[row])
        labels.append(y[row])
    if y[row] == 5:
        dataset.append(X.iloc[row])
        labels.append(y[row])
    if y[row] == 6:
        dataset.append(X.iloc[row])
        labels.append(y[row])
    if y[row] == 7:
        dataset.append(X.iloc[row])
        labels.append(y[row])
    if y[row] == 8:
        dataset.append(X.iloc[row])
        labels.append(y[row])


# In[80]:


Counter(labels)


# In[85]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[86]:


def feature_selection(train_instances):
    print('Crossvalidation started... ')
    selector = VarianceThreshold()
    selector.fit(train_instances)
    print('Number of features used... ' + str(Counter(selector.get_support())[True]))
    print('Number of features ignored... ' +str(Counter(selector.get_support())[False]))
    return selector


# In[87]:


#Learn the features to filter from train set
print("Selecting features... ")
fs = feature_selection(X_scaled)
 
#Transform train and test subsets
train_instances = fs.transform(X_scaled)


# In[89]:


sm = SMOTE(ratio = {5:2000, 11:2000, 10:2000, 6:2000, 7:2000, 4:2000, 8:2000}, random_state=42, k_neighbors = 2)
X_res, y_res = sm.fit_sample(train_instances, y)


# In[90]:


Counter(y_res)


# In[91]:


def evaluate(classifier, training_instances, training_labels):
    metrics = cross_validate(classifier, training_instances, training_labels, cv=10, 
                             n_jobs=-1, scoring=['accuracy'])
    print("Accuracy: %0.4f (+/- %0.4f)" % (metrics['test_accuracy'].mean(), metrics['test_accuracy'].std() * 2))
    print("Mean fit time: %0.4f ms" % (metrics['fit_time'].mean()))


# In[101]:


voting = VotingClassifier([ 
    ('knn', OneVsRestClassifier(KNeighborsClassifier(n_neighbors = 4))),
    ('et', OneVsRestClassifier(ExtraTreesClassifier(n_estimators = 1000))),
    ('xgboost', XGBClassifier(learning_rate =0.1, n_estimators=1000, num_class = 12,
                             min_child_weight=1, gamma=0,subsample=0.8,colsample_bytree=0.8,
                             objective= 'multi:softmax', nthread=4,scale_pos_weight=1,
                             seed=27, early_stopping_rounds=70, verbose=False)),
    ('randomforest', RandomForestClassifier(n_estimators=1000)),
    ('ada', AdaBoostClassifier(n_estimators=1000))], voting='soft', weights =[1,2,1,2,1])


# In[102]:


X_test_scaled = scaler.transform(Xtestfinal)
test_instances = fs.transform(X_test_scaled)


# In[103]:


voting.fit(X_res, y_res)


# In[104]:


predictions = voting.predict(test_instances)


# In[105]:


Counter(predictions)


# In[106]:


predictionnew = predictions


# In[107]:


knn = KNeighborsClassifier(n_neighbors = 4)
rf = RandomForestClassifier(n_estimators = 1000)


# In[98]:


knn.fit(X_res, y_res)
rf.fit(X_res, y_res)


# In[99]:


knnp = knn.predict(test_instances)
rfp = rf.predict(test_instances)


# In[108]:


newlist = []
get_indexes = lambda knnp, xs: [i for (y, i) in zip(xs, range(len(xs))) if knnp == y]
newlist.append(get_indexes(10, knnp))
for item in newlist:
    predictionnew[item] = 10
newlist = []
get_indexes = lambda rfp, xs: [i for (y, i) in zip(xs, range(len(xs))) if rfp == y]
newlist.append(get_indexes(7, rfp))
for item in newlist:
    predictionnew[item] = 7
Counter(predictionnew)


# In[109]:


np.savetxt('predictions.dat', predictionnew, delimiter=" ", fmt="%s")


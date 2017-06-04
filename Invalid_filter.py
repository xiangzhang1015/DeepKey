print(__doc__)
import tensorflow as tf
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import scipy.io as sc
import numpy as np
import xgboost as xgb
import pywt
import random
import pandas as pd
import time
from sklearn import preprocessing
from sklearn import svm
from sklearn.ensemble import IsolationForest
import scipy.io as sc
import numpy as np
from sklearn.externals import joblib

##/home/xiangzhang/matlabwork/
feature = sc.loadmat("/home/xiangzhang/matlabwork/AR_ID_8person.mat")
input_data = feature['AR_ID_8person']
print input_data.shape
n_sample=20000

r,col=input_data.shape
AR_feature_true=input_data[0:n_sample*6,0:col]
AR_feature_false=input_data[n_sample*6:n_sample*8,0:col]

count_FF=0
count_FT=0
count_TF=0
count_TT=0


clf = svm.OneClassSVM(nu=0.15, kernel="rbf", gamma=0.1)
clf.fit(AR_feature_true)
seg_number=200
a=time.clock()
# clf = joblib.load('AR_outlier_model.pkl')
# Outlier  input
for i in range(1000):
    # name=eval('test_'+str(i))
    id=np.random.randint(0,n_sample*2,size=1)
    name=AR_feature_false[id:id+seg_number]
    #  fit the model

    y_pred_outliers = clf.predict(name)
    predict=np.mean(y_pred_outliers)
    if predict>0:
        count_FT=count_FT+1
    else:
        count_FF=count_FF+1

#True input
# clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
# clf.fit(AR_feature_true)
for j in range(1000):
    # name=eval('test_'+str(i))
    id=np.random.randint(0,n_sample*6,size=1)
    name=AR_feature_true[id:id+seg_number]
    #  fit the model

    y_pred_outliers = clf.predict(name)
    predict=np.mean(y_pred_outliers)
    if predict>0:
        count_TT=count_TT+1
    else:
        count_TF=count_TF+1


print "this is EEG data, model is saved as EEG_outlier_model_2000.pkl"


print "count_FT", count_FT
print "count_FF",count_FF
print "count_TT", count_TT
print "count_TF",count_TF
b=time.clock()
print "run time", b-a
# joblib.dump(clf, 'EEG_outlier_model_2000.pkl') #save SVM model, the fitting is dramatically time-consumming
# clf = joblib.load('filename.pkl') ##This is how to load the saved svm model


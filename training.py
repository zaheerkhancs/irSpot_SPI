# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 07:52:46 2018

@author: zaheer
"""
import torchlight as tl
import numpy
import pandas
import os
import pandas as pd
from keras import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from matplotlib import pyplot
from mlxtend.plotting import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from psenac import PCPseTNC
import numpy as np
from sklearn.metrics import roc_curve,auc,precision_recall_curve,f1_score,average_precision_score,confusion_matrix, \
    cohen_kappa_score,matthews_corrcoef
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.layers import Dropout
from sklearn.preprocessing import LabelEncoder,normalize
import time
from ifeature import dlbl, \
    dtgrmListing, \
    apndInstances, \
    MakePattern, \
    InsertgappedVect, \
    InsertRCF, \
    MakeTestVector, \
    one_hot_encoding, \
    baseline_model, \
    getStart, \
    getStop, \
    PlotMe,\
    ttl_plotme, \
    plotme_precisionRecallCurve, \
    GetROC_Curve, \
    perf_measure, Import_data_labels


pc_psetnc = PCPseTNC(lamada=2, w=0.05)
from util import normalize_index
phyche_index = [
        [7.176, 6.272, 4.736, 7.237, 3.810, 4.156, 4.156, 6.033, 3.410, 3.524, 4.445, 6.033, 1.613, 5.087, 2.169, 7.237,
         3.581, 3.239, 1.668, 2.169, 6.813, 3.868, 5.440, 4.445, 3.810, 4.678, 5.440, 4.156, 2.673, 3.353, 1.668, 4.736,
         4.214, 3.925, 3.353, 5.087, 2.842, 2.448, 4.678, 3.524, 3.581, 2.448, 3.868, 4.156, 3.467, 3.925, 3.239, 6.272,
         2.955, 3.467, 2.673, 1.613, 1.447, 3.581, 3.810, 3.410, 1.447, 2.842, 6.813, 3.810, 2.955, 4.214, 3.581, 7.176]
    ]

vec = pc_psetnc.make_pcpsetnc_vec(open('data/testdata/xtestPred.fasta'), phyche_index=['Dnase I', 'Nucleosome'],
                                      extra_phyche_index=normalize_index(phyche_index, is_convert_dict=True))
vec= np.array(vec)
XtestPred = MakeTestVector()
XtestPred=np.array(XtestPred)
MergeVect= numpy.append(XtestPred,vec)
#XtestPred = sklearn.preprocessing(XtestPred)
MergeVect=MergeVect.transpose()
MergeVect = MergeVect[:(len(MergeVect))]
if (MergeVect.ndim == 1):
    xpredin = numpy.array([MergeVect])
X, y = Import_data_labels()
from sklearn.utils import shuffle
X, y = shuffle(X, y)
#X = sklearn.preprocessing.scale(X)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Fit the model

clf = baseline_model()
history= clf.fit(x_train , y_train , epochs=100,   batch_size=32)
#dnsModel.evaluate(x_test,y_test,verbose=1)
score, acc = clf.evaluate(x_test, y_test, batch_size=32)
print('Test score:', score)
print('Test accuracy:', acc)
probs = clf.predict_proba(x_test)
#y_test= clf.predict(xpredin)
probs = probs[:, -1]
# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, probs)
roc_auc = auc(fpr,tpr)
print("Area Under Curve: ", roc_auc)
GetROC_Curve(fpr, tpr, roc_auc)
# predict class values
yhat = clf.predict(x_test)
precision, recall, thresholds = precision_recall_curve(yhat.round(), probs)
 # calculate average precision score
ap = average_precision_score(y_test, probs)
#plotme_precisionRecallCurve(y_test, yhat)
#PlotMe(history)
#ttl_plotme(history)
f1 = f1_score(y_test, probs.round())
kappa = cohen_kappa_score(y_test,yhat.round())
#auc = auc(recall, precision).__float__()
cm = confusion_matrix(y_test, yhat.round()).ravel()
tn, fp, fn, tp = confusion_matrix(y_test, yhat.round()).ravel()
print("Confusion Matrix: ", cm)
print("Accuracy", acc)
print("Sensitivity: ", tp / (tp + fn))
print("specificity: ", tn / (tn + fn))
print("F1 Measure : ", f1)
print("Kappa Statistics", kappa)
print("MCC:", matthews_corrcoef(y_test, yhat.round()))
print("Area Under Curve: ", roc_auc)
print("pos_pred_val: ", tp / (tp + fp))
print("neg_pred_val: ", tn / (tn + fn))
""" Ploting Confusion Matrix"""


# save the model to disk
model_json = clf.to_json()
with open("model/model.json","w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
clf.save_weights("model/model.h5")
print("Saved model to disk")
# later...
# load json and create model
json_file = open('model/model.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
#result = loaded_model.predict(x_test,y_test)
result = loaded_model.predict(xpredin)
print("Accurac:disk model=",result)

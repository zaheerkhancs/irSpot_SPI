# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 07:52:46 2018

@author: zaheer
"""
import numpy
import pandas
import os
import pandas as pd
from keras import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from psenac import PCPseTNC
import numpy as np
from sklearn.metrics import roc_curve,auc,accuracy_score
import matplotlib.pyplot as plt
from keras.layers import Dropout
from sklearn.preprocessing import LabelEncoder
import time
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_curve,auc,precision_recall_curve,f1_score,average_precision_score,confusion_matrix

def Import_data_labels():
    a, label = dlbl('data/features001.csv', 5)
    b, label = dlbl('data/features002.csv', 12)
    c, label = dlbl('data/features003.csv', 66)
    aa = dtgrmListing(np.c_[a], len(label))
    bb = dtgrmListing(np.c_[b], len(label))
    cc = dtgrmListing(np.c_[c], len(label))
    X = []
    X = apndInstances(X, aa, len(label))
    X = apndInstances(X, bb, len(label))
    X = apndInstances(X, cc, len(label))
    X = np.array(X)
    label = one_hot_encoding(label)

    y = label
    return X, y

def acc(y_test, dltpred):
    return accuracy_score(y_test, dltpred)
def getStart():
    starttime=time()
    print("Starting Time.",starttime)
def getStop():
    endtime=time()
def plotme_precisionRecallCurve(ytest, yhat):
    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt
    from sklearn.utils.fixes import signature
    precision,recall,_ = precision_recall_curve(ytest,yhat)
    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall,precision,color='b',alpha=0.2,
             where='post')
    plt.fill_between(recall,precision,alpha=0.2,color='b',**step_kwargs)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0,1.05])
    plt.xlim([0.0,1.0])
    plt.title('Precision-Recall curve:',(average_precision_score))
def PlotMe(history):
    plt.plot(history.history['acc'])
    #plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('accuracy.png')
    plt.show()
    plt.clf()

    plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss.png')
    plt.show()
    plt.clf()
def ttl_plotme(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])  # RAISE ERROR
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train','Test'],loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])  # RAISE ERROR
    plt.title('Model Loss Evaluate')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train','Test'],loc='upper left')
    plt.show()
    plt.savefig('ttlPlot.png')
def GetROC_Curve(fpr,tpr, roc_auc):
    plt.figure()
    lw = 2
    plt.plot(fpr,tpr,color='red',
             lw=lw,label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0,1],[0,1],color='navy',lw=lw,linestyle='--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Positive Test ROC')
    plt.legend(loc="lower right")
    plt.show()
def CMCmapPlotme(cm):
    classes = ['Hot','Cold']
    plt.imshow(cm,interpolation='nearest',cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=45)
    plt.yticks(tick_marks,classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i,j in numpy.itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,format(cm[i,j],fmt),
                 horizontalalignment="center",
                 color="white" if cm[i,j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
def perf_measure(ytest,y_hat):
    yhat = y_hat.round()
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(yhat)):
        if ytest[i] == y_hat[i] == 1:
            TP += 1
        if yhat[i] == 1 and ytest[i] != yhat[i]:
            FP += 1
        if ytest[i] == yhat[i] == 0:
            TN += 1
        if yhat[i] == 0 and ytest[i] != yhat[i]:
            FN += 1

    return (TP,FP,TN,FN)

def dlbl(filename_,feature_):
    dataframe = pd.read_csv(filename_, header=0,sep=',')
    data = dataframe[dataframe.columns[0:int(feature_)]]
    label = dataframe['Label'].values.tolist()
    return data, label
def dtgrmListing(dtgramone, datagram_len):
    XSeqList_ = []
    for inst_indx in range(0,datagram_len):
        XSeqList_.append(dtgramone[inst_indx].tolist())
    return XSeqList_
def apndInstances(X_1,X_2,length):
    if (len(X_1) == 0):
        for inst_indx in range(0,length):
            X_1.append(X_2[inst_indx])
    else:
        for inst_indx in range(0,length):
            X_1[inst_indx] = np.append(X_1[inst_indx],X_2[inst_indx]).tolist()
    return X_1
def MakePattern(in_file):
    with open(in_file) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    content = content[1:]
    pattern = []
    for i in range(0,len(content)):
        for j in range(0,len(content[i])):
            if (content[i][j] != ' '):
                pattern.append(content[i][j])
    return pattern
def InsertgappedVect(feature,data):
    L3 = len(data) - 5 + 1
    L4 = len(data) - 6 + 1
    L5 = len(data) - 7 + 1
    L6 = len(data) - 8 + 1
    count_1 = 0;
    for i in range(0,L3):
        if (data[i] == 'T' and data[i + 4] == 'A'):
            count_1 += 1
    feature.append(count_1 / L3)
    count_1 = 0
    count_2 = 0
    for i in range(0,L4):
        if (data[i] == 'C' and data[i + 5] == 'C'):
            count_1 += 1
        elif (data[i] == 'G' and data[i + 5] == 'C'):
            count_2 += 1
    feature.append(count_1 / L4)
    feature.append(count_2 / L4)
    count_1 = 0;
    for i in range(0,L5):
        if (data[i] == 'C' and data[i + 6] == 'C'):
            count_1 += 1
    feature.append(count_1 / L5)
    count_1 = 0;
    for i in range(0,L6):
        if (data[i] == 'G' and data[i + 7] == 'G'):
            count_1 += 1
    feature.append(count_1 / L6)
    return feature
def InsertRCF(feature,data):
    reverse_complements_4 = ['CGCC','CTAA','GGCG','TTAG']
    reverse_complements_5 = ['AAAAG','CTTTT','AGATA','TATCT','CCCAC','GTGGG','CGCAC','GTGCG','CTAAG','CTTAG','GGCAC',
                             'GTGCC','GGCCA','TGGCC','TATAA','TTATA','TATCA','TGATA','TATGA','TCATA']
    L4 = len(data) - 4 + 1
    L5 = len(data) - 5 + 1

    count_1 = 0
    count_2 = 0

    for i in range(0,len(data) - 4 + 1):
        pattern = []
        for j in range(0,4):
            pattern.append(data[i + j])
        s = ''.join(pattern)

        if (s in reverse_complements_4):
            if (s == 'CGCC' or s == 'CTAA'):
                count_1 += 1
            else:
                count_2 += 1
    feature.append(count_1 / L4)
    feature.append(count_2 / L4)

    count_1 = 0
    count_2 = 0
    count_3 = 0
    count_4 = 0
    count_5 = 0
    count_6 = 0
    count_7 = 0
    count_8 = 0
    count_9 = 0
    count_10 = 0

    for i in range(0,len(data) - 5 + 1):
        pattern = []
        for j in range(0,5):
            pattern.append(data[i + j])
        s = ''.join(pattern)

        if (s in reverse_complements_5):
            if (s == 'AAAAG' or s == 'CTTTT'):
                count_1 += 1
            elif (s == 'AGATA' or s == 'TATCT'):
                count_2 += 1
            elif (s == 'CCCAC' or s == 'GTGGG'):
                count_3 += 1
            elif (s == 'CGCAC' or s == 'GTGCG'):
                count_4 += 1
            elif (s == 'CTAAG' or s == 'CTTAG'):
                count_5 += 1
            elif (s == 'GGCAC' or s == 'GTGCC'):
                count_6 += 1
            elif (s == 'GGCCA' or s == 'TGGCC'):
                count_7 += 1
            elif (s == 'TATAA' or s == 'TTATA'):
                count_8 += 1
            elif (s == 'TATCA' or s == 'TGATA'):
                count_9 += 1
            else:
                count_10 += 1

    feature.append(count_1 / L5)
    feature.append(count_2 / L5)
    feature.append(count_3 / L5)
    feature.append(count_4 / L5)
    feature.append(count_5 / L5)
    feature.append(count_6 / L5)
    feature.append(count_7 / L5)
    feature.append(count_8 / L5)
    feature.append(count_9 / L5)
    feature.append(count_10 / L5)

    return feature
def GetTestvect(data_):
    data = MakePattern(data_)
    feature = []
    feature = InsertgappedVect(feature, data)
    feature = InsertRCF(feature, data)
    return feature
def MakeTestVector():
    data = MakePattern("data/testdata/xtestPred.fasta")
    feature = []
    feature = InsertgappedVect(feature, data)
    feature = InsertRCF(feature, data)
    return feature
def one_hot_encoding(labels):
    encoder = LabelEncoder()
    encoder.fit(labels)
    encode_labels = encoder.transform(labels)
    return encode_labels
def baseline_model():
    dnsModel = Sequential()
    dnsModel.add(Dense(50,input_dim=83,kernel_initializer='uniform',activation='relu'))
    dnsModel.add(Dense(20,kernel_initializer='uniform',activation='relu'))
    dnsModel.add(Dropout(0.2, input_shape=(20,)))
    dnsModel.add(Dense(20,kernel_initializer='uniform',activation='relu'))
    dnsModel.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))
    dnsModel.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return dnsModel
"""
 #Customize Baseline Model...
"""
def Custom_baseline_model(lrrate):
    dnsModel = Sequential()
    Opt_=SGD(lr=0.0, momentum=0.9, decay=0.0, nesterov=False)
    dnsModel.add(Dense(50,input_dim=83,kernel_initializer='uniform',activation='relu'))
    dnsModel.add(Dense(20,kernel_initializer='uniform',activation='relu'))
    dnsModel.add(Dropout(0.2, input_shape=(20,)))
    dnsModel.add(Dense(20,kernel_initializer='uniform',activation='relu'))
    dnsModel.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))
    dnsModel.compile(loss='binary_crossentropy',optimizer=Opt_, metrics=['accuracy'])
    return dnsModel
def step_ecay(epoch):
    initial_lrate=0.1
    drop=0.5
    epoch_drop=10
    lrate= initial_lrate * numpy.math.pow(drop, numpy.math.floor((1 + epoch) / epoch_drop))
    return lrate
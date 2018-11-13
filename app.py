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
from matplotlib import pyplot
from mlxtend.plotting import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from psenac import PCPseTNC
import numpy as np
from sklearn.metrics import roc_curve,auc,precision_recall_curve,f1_score,average_precision_score,confusion_matrix, \
    cohen_kappa_score,matthews_corrcoef
import matplotlib.pyplot as plt
from flask import Flask,render_template,url_for,request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib

from sklearn import svm,datasets
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
    PlotMe, \
    ttl_plotme, \
    plotme_precisionRecallCurve, \
    GetROC_Curve,\
    perf_measure, \
    GetTestvect

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():

    if request.method == 'POST':
        import time
        ts = time.time()
        comment = request.form['comment']
        BASE_FOLDER="data"
        OUTPUT_FOLDER = "testdata"
        fcount = len(next(os.walk("./data/testdata")))
        PROJECT_ROOT_DIR = "."
        opath = os.path.join(PROJECT_ROOT_DIR, BASE_FOLDER, OUTPUT_FOLDER, str(int(fcount)+1)+"-"+str(ts)+".fasta")
        file = open(opath, "w+")
        file.write(comment)
        file.close()
        #data = [comment]
        pc_psetnc = PCPseTNC(lamada=2,w=0.05)
        from util import normalize_index
        phyche_index = [
            [7.176,6.272,4.736,7.237,3.810,4.156,4.156,6.033,3.410,3.524,4.445,6.033,1.613,5.087,2.169,7.237,
             3.581,3.239,1.668,2.169,6.813,3.868,5.440,4.445,3.810,4.678,5.440,4.156,2.673,3.353,1.668,4.736,
             4.214,3.925,3.353,5.087,2.842,2.448,4.678,3.524,3.581,2.448,3.868,4.156,3.467,3.925,3.239,6.272,
             2.955,3.467,2.673,1.613,1.447,3.581,3.810,3.410,1.447,2.842,6.813,3.810,2.955,4.214,3.581,7.176]
        ]
        #vec = pc_psetnc.make_pcpsetnc_vec(open('data/testdata/xtestPred.fasta'), phyche_index=['Dnase I','Nucleosome'], extra_phyche_index = normalize_index(phyche_index,is_convert_dict=True))
        vec = pc_psetnc.make_pcpsetnc_vec(open(opath), phyche_index=['Dnase I','Nucleosome'], extra_phyche_index =normalize_index(phyche_index,is_convert_dict=True))
        vec = np.array(vec)
        XtestPred = GetTestvect(opath)
        XtestPred = np.array(XtestPred)

        MergeVect = numpy.append(XtestPred,vec)
        # XtestPred = sklearn.preprocessing(XtestPred)
        MergeVect = MergeVect.transpose()
        MergeVect = MergeVect[:(len(MergeVect))]
        if (MergeVect.ndim == 1):
            xpredin = numpy.array([MergeVect])
        # later...
        # load json and create model
        json_file = open('model/model.json','r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model/model.h5")
        # print("Loaded model from disk")
        # result = loaded_model.predict(x_test,y_test)
        #vect = cv.transform(data).toarray()
        result = loaded_model.predict(xpredin)
        mypredict = loaded_model.predict_proba(xpredin)
        mypredict =numpy.asarray(result[0])
        my_pred = mypredict[0]
        predRaw = my_pred.round()
        return render_template('result.html', prediction=predRaw)
if __name__ == '__main__':
    app.run(debug=True)

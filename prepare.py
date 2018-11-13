# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 07:52:46 2018

@author: zaheer
"""
import pandas as pd
import numpy as np
class prepare:
    def _getData(datapath,filename,num_features):
        df = pd.read_csv(datapath + filename,header=0,sep=',')
        data_ = dataframe[dataframe.columns[0:int(num_features)]]
        fLabel_ = dataframe['Label'].values.tolist()
        return data_ , fLabel_

    def datagram_to_list(datagram,datagram_len):
        X_ = []
        for inst_indx in range(0,datagram_len):
            X_.append(datagram[inst_indx].tolist())
        return X_
    def append_instances(X_1,X_2,length):
        if (len(X_1) == 0):
            for inst_indx in range(0,length):
                X_1.append(X_2[inst_indx])
        else:
            for inst_indx in range(0,length):
                X_1[inst_indx] = np.append(X_1[inst_indx],X_2[inst_indx]).tolist()
        return X_1
    def raw_data_to_patterns(in_file):
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
    def include_gapped_feature(feature,data):
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
    def include_reverse_complement_features(feature,data):
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
    def test_feature_extract():
        data = raw_data_to_patterns(input_datapath)
        feature = []
        feature = include_gapped_feature(feature,data)
        feature = include_reverse_complement_features(feature,data)
        return feature

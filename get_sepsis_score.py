#!/usr/bin/env python
from sklearn.externals import joblib
import numpy as np

def get_sepsis_score(data, model):
    s_m = np.load('septic_mean.npy', allow_pickle=True)
    ns_m = np.load('Nonseptic_mean.npy', allow_pickle=True)
    data = np.delete(data, [34, 35, 36, 37, 38], 1)
    s_m = np.delete(s_m, [34, 35, 36, 37, 38])
    ns_m = np.delete(ns_m, [34, 35, 36, 37, 38])
    All = np.vstack((s_m, ns_m))
    maenAll = np.mean(All, axis=0)
    flag = data[-1, -1]

    for i in range(35):
    # for i in range(40):
        for j in range(np.size(data, 0)):
            if np.isnan(data[j, i]):
                if j>0:
                    before = data[0:(j),i]
                    before = before[~np.isnan(before)]
                else:
                    before = []
                if j<np.size(data, 0):
                    after = data[(j+1):(np.size(data, 0)), i]
                    after = after[~np.isnan(after)]
                else:
                    after = []

                if np.size(before)>0:
                    data[j, i] = before[-1]
                elif np.size(after)>0:
                    data[j, i] = after[0]

    #################### PART 2
    if flag == 1:
        mean = maenAll
    else:
        mean = maenAll

    for k in range(np.size(data, 0)):
        for h in range(35):
        # for h in range(40):
            if np.isnan(data[k, h]):
                data[k, h] = mean[h]



    M1 = joblib.load('model-saved.pkl')
    # data = np.nan_to_num(data)

    # W = np.load('W.npy')
    # m = np.load('m.npy')
    # data = np.dot(data - m, W.transpose())
    predicted = M1.predict(data)

    score = np.random.rand(len(data), 1)
    for i in range(len(data)):
        if predicted[i]==0:
         score[i] = 0.4
        else:
         score[i] = 0.6

    label = np.copy(predicted)

    return score, label

def load_sepsis_model():

    return None

# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import time
import seaborn as sns
import unicodedata
import nltk
from nltk.corpus import stopwords
from dask.distributed import Client
import joblib
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression
import lightgbm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from keras import models
from keras import layers
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import cross_validate, KFold, StratifiedKFold, cross_val_score

# nltk.download("stopwords")
stopwords_ = set(stopwords.words("french"))

def strip_accents(s):
   return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn')

def remove_stopwords(s, sw):
    tokens = s.split()
    clean_tokens = [t for t in tokens if not t in sw]
    clean_text = " ".join(clean_tokens)
    return clean_text

def remove_short(s):
    tokens = s.split()
    clean_tokens = [t for t in tokens if len(t) > 1]
    clean_text = " ".join(clean_tokens)
    return clean_text

# def daskrun(MLmethod):
#     daskop = Client(n_workers=4)
#     with joblib.parallel_backend('dask'):
#         MLmethod.fit(train_features, np.ravel(train_targets))
#     print(MLmethod.score(test_features, test_targets))
#
# def daskreport(MLmethod):
#     daskop = Client(n_workers=4)
#     with joblib.parallel_backend('dask'):
#         MLmethod.fit(train_features, np.ravel(train_targets))
#         predict = MLmethod.predict(test_features)
#         print(classification_report(test_targets, predict))


def get_text_classification(estimator, trainf, traint, testf, testt):

    start = time.time()
    model = estimator
    daskop = Client(n_workers=4)
    with joblib.parallel_backend('dask'):
        model.fit(trainf, np.ravel(traint))
        print(model)

        pred = model.predict(testf)
        print(pred)

        score = metrics.accuracy_score(testt, pred)
        matrix = metrics.confusion_matrix(testt, pred)
        report = metrics.classification_report(testt, pred)

    print('>>>Accuracy\n', score)
    print('\n>>>Recall\n', report)

    end = time.time()
    t = end - start
    print('\n>>>Duration：', t, 's\n')
    classifier = str(model).split('(')[0]

    return pred, classifier, score, round(t, 2), matrix, report

# def daskrunx(MLmethod, trainf, traint, testf, testt):
#     daskop = Client(n_workers=4)
#     with joblib.parallel_backend('dask'):
#         MLmethod.fit(trainf, np.ravel(traint))
#     print(MLmethod.score(testf, testt))
#
# def daskreportx(MLmethod, trainf, traint, testf, testt):
#     daskop = Client(n_workers=4)
#     with joblib.parallel_backend('dask'):
#         MLmethod.fit(trainf, np.ravel(traint))
#         predict = MLmethod.predict(testf)
#         print(classification_report(testt, predict))

df= pd.read_csv("data/valid_set.csv.gz", encoding = "utf-8", sep = ";", low_memory = False)

df.drop('Libellé.Prescription', axis=1, inplace=True)

dftemp = pd.read_csv("./dftemp.csv", encoding = "utf-8", sep = ",", low_memory = False)
dftemp.dropna(inplace=True)



df2 = pd.concat([df, pd.DataFrame(columns=['classif'])])
df2.columns = ['commentaire', 'classif']

# # print(df.columns)

#
# temp = df2['PLT']
#
# df2['PLT']= df2['PLT'].astype('str')
# for index, row in df2.iterrows():
#         PLT = row['PLT']
#         gravite = row['gravite']
#         catego = row['catego']
#
#         if PLT == '10.0' or PLT == '11.0':
#             df2.loc[index, 'catego'] = PLT[0]+PLT[1]
#         else:
#             df2.loc[index, 'catego'] = PLT[0]
#
#         if PLT.startswith('4') or PLT.startswith('5'):
#             df2.loc[index, 'gravite'] = 1
#         elif PLT == '6.3' or PLT == '6.4':
#             df2.loc[index, 'gravite'] = 1
#         else:
#             df2.loc[index, 'gravite'] = 0
#
dfpreQ2 = pd.DataFrame(columns=['comm'])
dfpreQ2['comm'] = df2['commentaire']

dfpreQ2.dropna(inplace=True)
dfpreQ2['comm'] = dfpreQ2['comm'].apply(lambda x: remove_stopwords(x, stopwords_))
stword1 = ['d\'']
dfpreQ2['comm'] = dfpreQ2['comm'].apply(lambda x: remove_stopwords(x, stword1))
dfpreQ2['comm'] = dfpreQ2['comm'].apply(strip_accents)
dfpreQ2['comm'] = dfpreQ2['comm'].str.lower()
dfpreQ2['comm'] = dfpreQ2['comm'].str.replace(r'[^a-z\s]', '')
dfpreQ2['comm'] = dfpreQ2['comm'].str.replace(r'\s+', ' ')
dfpreQ2['comm'] = dfpreQ2['comm'].str.strip()
dfpreQ2['comm'] = dfpreQ2['comm'].apply(remove_short)
stword2 = ['mg', 'umoll', 'kg', 'min', 'cp', 'cpr', 'ug']
dfpreQ2['comm'] = dfpreQ2['comm'].apply(lambda x: remove_stopwords(x, stword2))
#
predata = dfpreQ2.iloc[:, dfpreQ2.columns == 'comm']
# pretarget = dfpreQ2.loc[:, dfpreQ2.columns == 'catego']

# tfidf = TfidfVectorizer(max_features=1000)
# features = tfidf.fit_transform(predata['comm']).toarray()

# my_load_model = models.load_model('./dl1_model.h5')
#
# final = my_load_model.predict(features, verbose=1)

dfinal = pd.read_csv("./tempres.csv", encoding = "utf-8", sep = ",", low_memory = False)
#
# df2['classif'] = dfinal['classif']
# df2.to_csv('./Defi3_Groupe3_Q2.csv', sep=',', index=None, encoding='utf-8')
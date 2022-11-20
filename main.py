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
from sklearn.feature_extraction.text import TfidfVectorizer
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

def daskrun(MLmethod):
    daskop = Client(n_workers=4)
    with joblib.parallel_backend('dask'):
        MLmethod.fit(train_features, np.ravel(train_targets))
    print(MLmethod.score(test_features, test_targets))

def daskreport(MLmethod):
    daskop = Client(n_workers=4)
    with joblib.parallel_backend('dask'):
        MLmethod.fit(train_features, np.ravel(train_targets))
        predict = MLmethod.predict(test_features)
        print(classification_report(test_targets, predict))


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

df= pd.read_csv("data/data_defi3.csv.gz", encoding = "utf-8", sep = ";", low_memory = False)

df.drop('Libellé.Prescription', axis=1, inplace=True)
df.columns = ['commentaire', 'PLT']
# print(df.columns)
df2 = pd.concat([df, pd.DataFrame(columns=['catego', 'gravite'])])

temp = df2['PLT']

df2['PLT']= df2['PLT'].astype('str')
for index, row in df2.iterrows():
        PLT = row['PLT']
        gravite = row['gravite']
        catego = row['catego']

        if PLT == '10.0' or PLT == '11.0':
            df2.loc[index, 'catego'] = PLT[0]+PLT[1]
        else:
            df2.loc[index, 'catego'] = PLT[0]

        if PLT.startswith('4') or PLT.startswith('5'):
            df2.loc[index, 'gravite'] = 1
        elif PLT == '6.3' or PLT == '6.4':
            df2.loc[index, 'gravite'] = 1
        else:
            df2.loc[index, 'gravite'] = 0

dfpreQ2 = pd.DataFrame(columns=['comm', 'catego'])
dfpreQ2['comm'] = df2['commentaire']
dfpreQ2['catego'] = df2['catego']

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

predata = dfpreQ2.iloc[:, dfpreQ2.columns == 'comm']
pretarget = dfpreQ2.loc[:, dfpreQ2.columns == 'catego']

tfidf = TfidfVectorizer(max_features=1000, min_df=5, max_df=0.7)
features = tfidf.fit_transform(predata['comm']).toarray()
targets = pretarget['catego']

train_features, test_features, train_targets, test_targets = train_test_split(features, targets,
                                                                              train_size=0.8,
                                                                              test_size=0.2,
                                                                              random_state=42,
                                                                              shuffle = True,
                                                                              stratify=targets
)

if __name__ == '__main__':
    estimator_list, score_list, time_list = [], [], []
    ### Random Forest
    rf = RandomForestClassifier(n_estimators=1000, random_state=42, verbose=1)
    resultrf = get_text_classification(rf, train_features, train_targets, test_features, test_targets)
    estimator_list.append(resultrf[1])
    score_list.append(resultrf[2])
    time_list.append(resultrf[3])

    ### MNB
    mnb = MultinomialNB()
    resultmnb = get_text_classification(mnb, train_features, train_targets, test_features, test_targets)
    estimator_list.append(resultmnb[1])
    score_list.append(resultmnb[2])
    time_list.append(resultmnb[3])

    ### GaussianNB
    gnb = GaussianNB()
    resultgnb = get_text_classification(gnb, train_features, train_targets, test_features, test_targets)
    estimator_list.append(resultgnb[1])
    score_list.append(resultgnb[2])
    time_list.append(resultgnb[3])

    ### Logistic Regression
    lr = LogisticRegression(random_state=42, solver='lbfgs', multi_class='multinomial', max_iter=1000)
    resultlr = get_text_classification(lr, train_features, train_targets, test_features, test_targets)
    estimator_list.append(resultlr[1])
    score_list.append(resultlr[2])
    time_list.append(resultlr[3])

    ### LightGBM
    lgbm = lightgbm.LGBMClassifier(random_state=42, n_estimators=1000)
    resultlgbm = get_text_classification(lgbm, train_features, train_targets, test_features, test_targets)
    estimator_list.append(resultlgbm[1])
    score_list.append(resultlgbm[2])
    time_list.append(resultlgbm[3])

    # ### Deep Learning 1
    # start = time.time()
    #
    # feature_num = train_features.shape[1]  # 设置所希望的特征数量
    #
    # train_targets_cate = to_categorical(train_targets)
    # test_targets_cate = to_categorical(test_targets)
    # #print(train_targets_cate)











    dfeva = pd.DataFrame()
    dfeva['model'] = estimator_list
    dfeva['accuracy'] = score_list
    dfeva['time/s'] = time_list
    print(dfeva)












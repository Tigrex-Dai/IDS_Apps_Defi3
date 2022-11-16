# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import seaborn as sns
import unicodedata
import nltk
from nltk.corpus import stopwords

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





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk import word_tokenize, wordpunct_tokenize
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from nltk.corpus import stopwords
# nltk.download('stopwords')
# nltk.download('punkt')


def read_data():
    df = pd.read_csv('train.csv')
    X, y = df.drop(columns=['target']), df[['target']]
    return X, y


def clean_data(X):
    stop = stopwords.words('english')
    X['text'] = X['text'].apply(lambda row: ' '.join([word for word in row.split() if word.lower()
                                                      not in stop and 'http' not in word]))
    X['text'] = X['text'].apply(lambda row: ' '.join([w for w in wordpunct_tokenize(row) if
                                                      w.isalnum() and len(w) > 1]))
    return X


def ML_Model(X, y):
    clf = SVC()
    clf.fit(X, y)
    print(clf.score(X, y))
    return clf


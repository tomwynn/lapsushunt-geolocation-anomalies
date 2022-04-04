import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

class MultiColumnLabelEncoder:

    def __init__(self, columns=None):
        self.columns = columns # array of column names to encode


    def fit(self, X, y=None):
        self.encoders = {}
        columns = X.columns if self.columns is None else self.columns
        for col in columns:
            self.encoders[col] = LabelEncoder().fit(X[col])
        return self


    def transform(self, X):
        output = X.copy()
        columns = X.columns if self.columns is None else self.columns
        for col in columns:
            output[col] = self.encoders[col].transform(X[col])
        return output


    def fit_transform(self, X, y=None):
        return self.fit(X,y).transform(X)


    def inverse_transform(self, X):
        output = X.copy()
        columns = X.columns if self.columns is None else self.columns
        for col in columns:
            output[col] = self.encoders[col].inverse_transform(X[col])
        return output

if __name__ == "__main__":
    df = pd.read_csv('redacted.csv')
    df = df.loc[:,df.columns.isin(['alternateId','clientCountry', 'clientIpAddress'])] 

    multi = MultiColumnLabelEncoder(columns=['alternateId','clientCountry', 'clientIpAddress'])
    X = multi.fit_transform(df)
    random_state = np.random.RandomState(42)
    clf = IsolationForest(n_estimators=100,max_samples='auto',contamination=float(0.2),random_state=random_state)
    preds = clf.fit_predict(X)
    X['anomaly'] = clf.fit_predict(X[['clientCountry']])
    anomaly=X.loc[X['anomaly']==-1]
    anomaly_index=list(anomaly.index)
    inv = multi.inverse_transform(anomaly)
    dedup = inv.drop_duplicates()
    print(dedup) 

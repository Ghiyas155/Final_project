#!/usr/bin/env python
# coding: utf-8

# # Student Name: Ghiyas Nizamudden Shaik

# # Instructor: Yasser Abduallah

# ## CS634 End Term Project: Implementation of Supervised Data Mining for Binary Classification

# ### Import libraries

# In[1]:


#import the necessery libraries

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

tf.random.set_seed(42)
pd.options.display.max_columns = None


# ### Import data

# In[2]:


# import data

train = pd.read_csv("data/airline_train.csv", index_col=0)
train = train.set_index("id")


# ### Fill missing values

# In[3]:


# fill the missing arrival delay with departure delay times assuming they will don't lose or gain time

train.loc[:, "Arrival Delay in Minutes"] = train.loc[:, "Arrival Delay in Minutes"].fillna(train.loc[:, "Departure Delay in Minutes"])


# ### Categorical Encoding

# In[4]:


# encode the categorical columns into numerical values

cat_encod = {
    "Gender": ['Male', 'Female'],
    "Customer Type": {'Loyal Customer': 1, 'disloyal Customer': -1},
    "Type of Travel": ['Personal Travel', 'Business travel'],
    "Class": {'Eco Plus': 1, 'Business': 2, 'Eco': 0},
    "satisfaction": {'neutral or dissatisfied':0, 'satisfied': 1}
    
}

for col, val in cat_encod.items():
    if type(val) == dict:
        train.loc[:, col] = train.loc[:, col].map(val)
        train[col] = train[col].astype(int)
        
    else:
        train.loc[:, val] = pd.get_dummies(train.loc[:, col])
        train[val] = train[val].astype(int)
        train = train.drop(col, axis=1)


# ### Split features and target variables

# In[5]:


# extract predictor and target variables

y = train["satisfaction"]
X = train.drop("satisfaction", axis=1)


# ### Data Transformation

# In[6]:


# scale the data

scaler = StandardScaler()
X = scaler.fit_transform(X)


# ### Evaluating Classifiers
# 
# TP, TN, FP, FN, TPR(sensitivity, r), TNR(specificity)<br>
# FPR, FNR, FDR, NPV, p, F1, acc, err<br>
# BACC, TSS, HSS, BS, BSS<br>

# In[7]:


# returns confusion matrix values given predicted and ground truth values

def confusion_matrix(preds, truth):
    tp = np.sum((preds == 1) & (truth == 1))
    fp = np.sum((preds == 1) & (truth == 0))
    tn = np.sum((preds == 0) & (truth == 0))
    fn = np.sum((preds == 0) & (truth == 1))
    return (tp, fp, tn, fn)


# In[8]:


# returns a dictionary of different metrics given the confusion matrix

def metrics(tp, fp, tn, fn):
    model_metrics = {}
    tpr = tp / (tp + fn)
    model_metrics["tpr"] = tpr
    tnr = tn / (fp + tn)
    model_metrics["tnr"] = tnr
    fpr = fp / (fp + tn)
    model_metrics["fpr"] = fpr
    fnr = fn / (tp + fn)
    model_metrics["fnr"] = fnr
    
    model_metrics["recall"] = tpr
    prec = tp / (tp + fp)
    model_metrics["precision"] = prec
    f1 = 2 * prec * tpr / (prec + tpr)
    model_metrics["f1-score"] = f1
    acc = (tp + tn) / (tp + fn + fp + tn)
    model_metrics["accuracy"] = acc
    err = (fp + fn) / (tp + fn + fp + tn)
    model_metrics["error rate"] = err
    npv = tn / (tn + fn)
    model_metrics["npv"] = npv
    fdr = fp / (fp + tp)
    model_metrics["fdr"] = fdr
    
    bacc = (tpr + tnr) / 2
    model_metrics["bacc"] = bacc
    tss = tpr - fpr
    model_metrics["tss"] = tss
    hss = 2 * (tp * tn - fp * fn) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    model_metrics["hss"] = hss

    return model_metrics


# In[9]:


metric_list=["tpr", "tnr", "fpr", "fnr", "recall", "precision", "f1-score", "accuracy", "error rate", "npv", "fdr",
                                    "bacc", "tss", "hss"]


# In[10]:


# use k-fold with 10 splits

kf = KFold(n_splits=10, shuffle=True, random_state=42)


# ### Random Forest

# In[11]:


rf_metrics = pd.DataFrame([], columns=metric_list)

for i, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X_train, y_train)
    pred_proba = rf.predict_proba(X_test)[:, 1]
    preds = (pred_proba >= 0.5).astype(int)

    tp, fp, tn, fn = confusion_matrix(preds, y_test)
    model_metrics = metrics(tp, fp, tn, fn)
    rf_metrics.loc[i+1,:] = model_metrics
    
rf_avg = rf_metrics.mean()


# In[12]:


print(rf_metrics)


# ### Deep Learning: LSTM

# In[13]:


lstm_metrics = pd.DataFrame([], columns=metric_list)

for i, (train_index, test_index) in enumerate(kf.split(X_train)):
    model = Sequential([
        Input(shape=(X_train.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(1, activation='sigmoid')
    ])

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    
    pred_proba = model.predict(X_test)
    preds = (pred_proba >= 0.5).astype(int)

    tp, fp, tn, fn = confusion_matrix(preds.reshape(-1), y_test)
    model_metrics = metrics(tp, fp, tn, fn)
    lstm_metrics.loc[i+1,:] = model_metrics
    
lstm_avg = lstm_metrics.mean()


# In[14]:


print(lstm_metrics)


# ### Algorithms: KNN

# In[15]:


knn_metrics = pd.DataFrame([], columns=metric_list)

for i, (train_index, test_index) in enumerate(kf.split(X_train)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    pred_proba = knn.predict_proba(X_test)[:, 1]
    preds = (pred_proba >= 0.5).astype(int)

    tp, fp, tn, fn = confusion_matrix(preds, y_test)
    model_metrics = metrics(tp, fp, tn, fn)
    knn_metrics.loc[i+1,:] = model_metrics

knn_avg = knn_metrics.mean()


# In[16]:


print(knn_metrics)


# In[17]:


# average of all metrics for the three models

final_df = pd.concat([rf_avg, lstm_avg, knn_avg], axis=1)
final_df.columns=["Random Forest", "LSTM", "KNN"]
print(final_df)


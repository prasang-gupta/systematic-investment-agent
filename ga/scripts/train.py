import pandas as pd
import numpy as np
from numpy import array
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import datetime as dt
import math
import warnings
import random as rnd
from datetime import datetime
from datetime import timedelta
import sys, os
import argparse

parser = argparse.ArgumentParser(description='Learn the policy behind optimal action using Neural Network architecture')
parser.add_argument('--etf', action='store', type=str, default='VTI',
                    help='index fund identifier (default: VTI)')
parser.add_argument('--end_date', action='store', type=str, default='2019-12-31',
                    help='period upto which training is considered (default: 2019-12-31)')
parser.add_argument('--epochs', action='store', type=int, default=150,
                    help='number of epochs for training (default: 150)')
args = parser.parse_args()

df = pd.read_csv('Modeling_data_for_'+args.etf+'_'+args.end_date+'_.csv')

df = df.copy().iloc[2:,1:]
df = df.copy().fillna(0)

# df = df.copy().drop(['vol_chg','vol_chg_5ma','vol_chg_10ma','vol_chg_30ma'],axis=1)

y = [] 
X = []

for sample in (df['sample_id'].unique()):
  
  try:
    loop_df = df[(df['sample_id'] == sample) & (df['state'] != -9999)].drop(['action','sample_id','pur_price','state'],axis=1).copy()
    # create scaler
    scaler = StandardScaler()
    # fit scaler on data
    # print(sample)
    scaler.fit(df[(df['sample_id'] == sample) & (df['state'] == -9999)].drop(['action','sample_id','pur_price','state'],axis=1))
    # apply transform
    norm_df = scaler.transform(loop_df)
    X.append(norm_df)
    
    chunk_y = df[(df['sample_id'] == sample) & (df['state'] != -9999)]['action'].values
    y.append(chunk_y)
  except:
    pass

y = np.concatenate(y,axis = 0)
X = np.concatenate(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3

    if epoch > 120:
        lr *= 1e-5
    elif epoch > 100:
        lr *= 1e-4
    elif epoch > 80:
        lr *= 1e-3
    elif epoch > 50:
        lr *= 1e-2
    elif epoch > 30:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

model = Sequential()

model.add(Dense(500, activation='relu'))  
model.add(Dropout(0.3))

model.add(Dense(250, activation='relu'))  
model.add(Dropout(0.25))

model.add(Dense(125, activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(60, activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(30, activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(15, activation='relu'))
model.add(Dropout(0.25))

# model.add(Dense(4, activation='relu'))
# model.add(Dropout(0.1))

model.add(Dense(1, activation = 'sigmoid'))

epochs = args.epochs
decay_rate = lr_schedule(epochs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# fit model
acc = model.fit(X_train, y_train, validation_split=0.25, epochs=epochs, verbose=1)
print(acc)

# serialize model to JSON
model_json = model.to_json()
with open("Neurips_agent_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save("Neurips_agent_model_CUSTOM_MODEL.h5")
print("Saved model to disk")


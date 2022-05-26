import pandas as pd
import numpy as np
import datetime as dt
import math
import warnings
import random as rnd
from datetime import datetime
from datetime import timedelta
import sys, os
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import yfinance as yfin
import pickle
import argparse

parser = argparse.ArgumentParser(description='Get the results for the trained GADLE model')
parser.add_argument('--path', action='store', type=str, default='pretrained', 
                    help='path where model outputs are stored (default: pretrained)')
parser.add_argument('--start_date', action='store', type=str, default='2020-1-1',
                    help='start date for validation period (default: 2020-1-1)')
args = parser.parse_args()

# Disable print function
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore print function
def enablePrint():
    sys.stdout = sys.__stdout__

blockPrint()

with open('../rawdata.pkl', 'rb') as handle:
  stock_df = pickle.load(handle)

dates = stock_df.index.unique()
dates_rel = stock_df[stock_df.index>args.start_date].index.unique()

model = load_model(os.path.join(args.path, 'Neurips_agent_model_CUSTOM_MODEL.h5'))


X=[]
df_for_price=[]
period_label = []
st_idx = len(dates) - len(dates[(dates > args.start_date)])
for i in range(math.ceil(len(dates_rel)/30)):
  try:
    loop_df = stock_df[(stock_df.index>=dates[st_idx+30*i]) & (stock_df.index < dates[st_idx+30*(i+1)])].copy()
    # create scaler
    scaler = StandardScaler()
    # fit scaler on data
    scaler.fit(stock_df[(stock_df.index>=dates[st_idx-30+30*i]) & (stock_df.index < dates[st_idx+30*(i)])].copy())
    # apply transform
    norm_df = scaler.transform(loop_df)
    X.append(norm_df)
    df_for_price.append(stock_df[(stock_df.index>=dates[st_idx+30*i]) & (stock_df.index < dates[st_idx+30*(i+1)])]['avg_price'].reset_index())
    period_label.append([np.min(stock_df[(stock_df.index>=dates[st_idx+30*i])].index)]*30)
  except:
    pass

X=np.concatenate(X)
predicted_x= np.concatenate((model.predict(X) > 0.5).astype("int32"))*1.

df_with_predictions  = pd.DataFrame(np.concatenate(df_for_price))
df_with_predictions['Action'] = predicted_x
df_with_predictions['Period_Start'] = np.concatenate(period_label)
df_with_predictions = df_with_predictions.rename(columns={0:'Date',1:'Price'}).copy()
df_with_predictions = df_with_predictions.set_index('Date').copy()
df_with_predictions['Price'] = pd.to_numeric(df_with_predictions['Price'])

df_summary = df_with_predictions[['Price','Period_Start']].groupby('Period_Start').mean()

df_summary['Agent_Price'] = df_with_predictions[df_with_predictions['Action']==1][['Price','Period_Start']].groupby('Period_Start').mean()
df_summary['Daily_Count'] = 30
df_summary['Agent_Count'] = df_with_predictions[df_with_predictions['Action']==1][['Price','Period_Start']].groupby('Period_Start').count()*2
df_summary = df_summary.rename(columns={'Price':'Daily_Price'}).copy()
df_summary['Agent_Return_Over_Daily'] = (1-(df_summary['Agent_Price']/df_summary['Daily_Price']))*100
df_summary['Agent_Count_Over_Daily'] = (df_summary['Agent_Count'] - df_summary['Daily_Count'])
summary_final = df_summary.sort_index()

enablePrint()

print("")
print("")
print("")
print("----------------------------------------------------------------------------")
print("----------------------Results from Validation of Model----------------------")
print("----------------------------------------------------------------------------")
print("")
print(summary_final)
print("")
a = round(np.sum(summary_final['Daily_Price'])/len(summary_final),2)
print('Average Daily Price is',a)
b = round(np.sum(summary_final['Agent_Price']*summary_final['Agent_Count'])/np.sum(summary_final['Agent_Count']),2)
print('Average Agent Price is',b)
c = round((1-(b/a))*100,2)
print('Total Agent Return over Daily Purchase is',c,'%') 
d = (np.sum(summary_final['Agent_Count']) - np.sum(summary_final['Daily_Count'])) / len(summary_final)
print('Monthly Average Agent Purchased Count over Daily Purchase Count is',d,'Units')                 
print("")
print("")

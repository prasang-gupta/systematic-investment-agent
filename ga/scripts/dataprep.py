import pandas as pd
import numpy as np
import random as rnd
from genetic_algo_package import geneticalgorithm as ga
import sys, os
import yfinance as yfin
import time
from multiprocessing import Pool
from itertools import repeat
import psutil
import argparse

parser = argparse.ArgumentParser(description='Generate episodic samples, perform feature engineering, scaling and find the optimal action vectors using GA')
parser.add_argument('--etf', action='store', type=str, default='VTI',
                    help='index fund identifier (default: VTI)')
parser.add_argument('--end_date', action='store', type=str, default='2019-12-31',
                    help='period upto which training is considered (default: 2019-12-31)')
parser.add_argument('--episodes', action='store', type=int, default=4245,
                    help='number of episodes to generate (default: 4245)')
args = parser.parse_args()

# Disable print function
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore print function
def enablePrint():
    sys.stdout = sys.__stdout__

#Download data from Yahoo Finance
def get_data():
    # stock_df = web.DataReader(ETF,'yahoo',start = '2000-1-1',end = END_DATE)
    stock_df = yfin.Ticker(args.etf).history(start='2000-1-1', end=args.end_date).sort_values(by='Date')

    #Feature engineering that will be used later
    stock_df['avg_price'] = stock_df[['High','Low','Open','Close']].mean(axis=1)
    stock_df['avg_price_5ma'] = stock_df['avg_price'].rolling(window=5).mean()
    stock_df['avg_price_10ma'] = stock_df['avg_price'].rolling(window=10).mean()
    stock_df['avg_price_30ma'] = stock_df['avg_price'].rolling(window=30).mean()
    stock_df['avg_price_60ma'] = stock_df['avg_price'].rolling(window=60).mean()
    stock_df['avg_price_100ma'] = stock_df['avg_price'].rolling(window=100).mean()
    stock_df['avg_price_180ma'] = stock_df['avg_price'].rolling(window=180).mean()
    stock_df['avg_price_360ma'] = stock_df['avg_price'].rolling(window=360).mean()

    stock_df['price_chg'] = stock_df['avg_price'].pct_change()*100
    stock_df['price_chg_5sum'] = stock_df['price_chg'].rolling(window=5).sum()
    stock_df['price_chg_10sum'] = stock_df['price_chg'].rolling(window=10).sum()
    stock_df['price_chg_30sum'] = stock_df['price_chg'].rolling(window=30).sum()

    stock_df = stock_df.copy().drop(['Close','High','Low','Open','Volume','Dividends','Stock Splits'],axis=1)
    stock_df = stock_df.copy().dropna().sort_values('Date')

    return stock_df

def environment_sampler(stock_df, inputlist, lengthsample):
    '''
    Sampler to take list of dates and return sample environment of a specific length
    '''
    sample_date = rnd.choice(inputlist)
    environment_sample = stock_df[(stock_df.index >= sample_date)].iloc[:30]
    environment_scaler = stock_df[(stock_df.index < sample_date)].iloc[-30:]
    return environment_sample, environment_scaler

# Create buy functions 
def buy_1():
    return 0

def buy_2():
    return 2

def pur_price(row):
    if float(row['action']) > 0:
        return float(row['action'])*float(row['avg_price'])

def run_ga(episodeidx, stock_df, list_of_dates):
    log_start = time.time()
    blockPrint()

    action_list = [buy_1(),buy_2()]
    environment, environment_scaler = environment_sampler(stock_df, inputlist=list_of_dates, lengthsample=30)

    #Define loss function to solve each episode
    def fitness(gene, env_price = environment['avg_price'].values):
        mp_env = np.mean(env_price)
        len_gene = np.sum(gene)
        pdt_gene_price = np.dot(gene,env_price) if len_gene else 0
        mkt_return = 10/1200
        size_gene = 30
        return ((((pdt_gene_price/len_gene) - mp_env)*2*len_gene) / mp_env)  + ((1-(len_gene/(size_gene/2))) ** 2)

    #configure parameters of genetic algorithm
    algorithm_param = {'max_num_iteration': 200,                  
    'population_size':100,                 
    'mutation_probability':0.2,               
        'elit_ratio': 0.3,                 
            'crossover_probability': 0.4,           
                    'parents_portion': 0.3,                
                        'crossover_type':'uniform',              
                            'max_iteration_without_improv':30}

    #solve each episode to arrive at optimal action, given the loss function
    model=ga(function=fitness,dimension=30,variable_type='bool',algorithm_parameters=algorithm_param)
    env_price = environment['avg_price'].values
    model.run()

    #store optimal action vector
    environment['action'] = model.output_dict['variable']

    #Create columns in dataset for Nueral Network Training
    environment['pur_price'] = environment.apply(lambda row: pur_price(row), axis=1)
    environment['state'] = environment['pur_price'].rolling(30, min_periods = 1).mean()
    environment['sample_id'] = episodeidx

    environment_scaler['sample_id'] = episodeidx
    environment_scaler['state'] = -9999

    enablePrint()
    log_end = time.time()
    print(f'Completed episode {episodeidx} in {round(log_end - log_start, 2)}s')

    return environment, environment_scaler


if __name__ == '__main__':

    stock_df = get_data()

    # Create funtions
    list_of_dates = stock_df.iloc[:-30].index.unique()
    
    pool = Pool(psutil.cpu_count())
    results = pool.starmap(run_ga, zip(range(args.episodes), repeat(stock_df), repeat(list_of_dates)))
    pool.close()
    pool.join()

    results = list(zip(*results))
    envr, sclr = list(results[0]), list(results[1])

    #Export Data for Neural Network Training
    data_compiled = pd.concat([pd.concat(envr),pd.concat(sclr)], axis = 0, ignore_index=True)
    data_compiled.to_csv('Modeling_data_for_'+args.etf+'_'+args.end_date+'_.csv')
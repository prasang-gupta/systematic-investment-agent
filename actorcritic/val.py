import matplotlib.pyplot as plt
import re
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from sklearn.preprocessing import StandardScaler
import argparse
import os

parser = argparse.ArgumentParser(description='Get results for Actor Critic agent training')
parser.add_argument('--path', action='store', type=str, default='pretrained', 
                    help='path where model outputs are stored (default: pretrained)')
parser.add_argument('--model', action='store', type=str, default='model_280000',
                    help='model file to get results for (default: latest file)')
args = parser.parse_args()

rr = []
ir = []
ep = []
br = []
r1 = []
r2 = []
with open(os.path.join(args.path, 'logs')) as f:
    for line in f.readlines():
        rr.append(float(re.findall(r'reward: [0-9.-]+', line)[0][8:]))
        ir.append(float(re.findall(r'reward [0-9.-]+', line)[0][7:]))
        ep.append(int(re.findall(r'episode [0-9.-]+', line)[0][8:]))
        br.append(float(re.findall(r'br [0-9.-]+', line)[0][3:]))
        try:
            r1.append(float(re.findall(r'rew_1: [0-9.-]+', line)[0][7:]))
            r2.append(float(re.findall(r'rew_2: [0-9.-]+', line)[0][7:]))
        except:
            pass

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

class TestEnvironment:
    def __init__(self, window=30):
        """Test Environment to run the agent on the 2020 dataset (out of train)

        Args:
            window (int, optional): Window size for the episode. Defaults to 30.
        """
        self.data = np.load(os.path.join(args.path, 'episode_2020_scaled.npy'))                 # Scaled data
        self.stock_data = np.load(os.path.join(args.path, 'episode_2020_prices.npy'))     # Original data
        self.window = window            # Window size

    def score(self, model):
        """Score the agent performance

        Args:
            model (keras_model): DQN model

        Returns:
            (float, float): Overall agent Return over Daily, Overall agent Purchase Count over Daily
        """
        months = len(self.data) // self.window          # Distribute the data in chunks of window size
        daily_prices = []
        agent_prices = []
        buy_ratios = []
        agent_rods = []
        agent_counts = []
        print('AgAvg\tDAvg\tRoD\tAgCnt\tPCoD\tBuyR')
        for month in range(months):
            agent_price, daily_price, buy_ratio, agent_count = self._score_month(month, model)       # Write prices and buy_ratio for every month
            agent_rod = (1-(agent_price/daily_price)) * 100
            agent_count *= 2
            daily_prices.append(daily_price)
            agent_prices.append(agent_price)
            buy_ratios.append(buy_ratio)
            agent_rods.append(agent_rod)
            agent_counts.append(agent_count)
            print('{:5.2f}\t{:5.2f}\t{:5.2f}%\t{:5d}\t{:5d}\t{:5.2f}'.format(agent_price, daily_price, agent_rod, agent_count, agent_count-self.window, buy_ratio))
        overall_daily_price = round(np.mean(daily_prices), 2)
        overall_agent_price = round(np.dot(agent_prices, agent_counts) / np.sum(agent_counts), 2)
        overall_agent_rod = round((1-(overall_agent_price/overall_daily_price)) * 100, 2)
        overall_agent_counts = np.sum(agent_counts)
        overall_agent_pcod = round((np.sum(agent_counts) - (months * self.window)) / months, 2)
        overall_buy_ratio = round(np.mean(buy_ratios), 2)
        print('Overall')
        print('{:5.2f}\t{:5.2f}\t{:5.2f}%\t{:5d}\t{:5.2f}\t{:5.2f}'.format(overall_agent_price, overall_daily_price, overall_agent_rod, overall_agent_counts, overall_agent_pcod, overall_buy_ratio))
        return overall_agent_rod, overall_agent_pcod

    def _score_month(self, month, model):
        """Calculate agent score for a month

        Args:
            month (int): Idx of the month for which score needs to be calculated
            model (keras_model): DQN model

        Returns:
            (float, float, float, int): Daily average price, Agent average price, Buy ratio and Total buy actions for the agent
        """
        data = self.data[month*self.window:(month+1)*self.window]               # Scaled data for the month
        price = self.stock_data[month*self.window:(month+1)*self.window]        # Original prices for the month
        actions = []
        idx = 0
        for state in data:                                  # Run the model for the whole month, one day at a time storing actions
            progress = (idx + 1) / self.window
            buy_ratio = actions.count(1) / len(actions) if len(actions) else 0
            state = np.append(state, [progress, buy_ratio]).reshape(1,-1)
            state = tf.convert_to_tensor(state)
            action_probs, critic_value = model(state)
            action = np.argmax(np.squeeze(action_probs))
            actions.append(action)
            idx += 1
        daily_price = price.mean()              # Average daily price
        agent_price = 0 if not actions.count(1) else np.dot(actions, price) / actions.count(1)          # Average agent price
        return agent_price, daily_price, actions.count(1) / len(actions), actions.count(1)

testenv = TestEnvironment()
model = keras.models.load_model(os.path.join(args.path, args.model))
testenv.score(model)
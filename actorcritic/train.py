import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import yfinance as yfin
import pandas as pd
from sklearn.preprocessing import StandardScaler
import argparse
import os

parser = argparse.ArgumentParser(description='Train Actor Critic agent')
parser.add_argument('--discount_rate', action='store', type=float, default=0.97,
                    help='discount rate for past rewards (default: 0.97)')
parser.add_argument('--lr_initial', action='store', type=float, default=0.001,
                    help='initial learning rate (default: 0.001)')
parser.add_argument('--lr_decay_steps', action='store', type=int, default=1000,
                    help='learning rate decay steps (default: 1000)')
parser.add_argument('--lr_decay_rate', action='store', type=float, default=0.99,
                    help='learning rate decay rate (default: 0.99)')
parser.add_argument('--lr_staircase', action='store', type=bool, default=True,
                    help='staircase for learning rate decay (default: True)')
parser.add_argument('--explore_rate', action='store', type=float, default=1.0,
                    help='epsilon (default: 1.0)')
parser.add_argument('--explore_min', action='store', type=float, default=0.01,
                    help='minimum epsilon for exploration (default: 0.01)')
parser.add_argument('--explore_decay', action='store', type=float, default=0.999,
                    help='decay factor for exploration rate (default: 0.999)')
parser.add_argument('--model_save_freq', action='store', type=float, default=10000,
                    help='save model after these many episodes (default: 10000)')
parser.add_argument('--seed', action='store', type=int, default=42,
                    help='seed for training (default: 42)')
args = parser.parse_args()

def make_episode_2020(window = 30):
    """Generate contextually scaled episode data for 2020

    Args:
        window (int, optional): Window size for the episode. Defaults to 30.
    """
    # Downloading data from yahoo finance
    stock_df = yfin.Ticker("VTI").history(start='2018-1-1', end='2020-12-31').sort_values(by='Date')

    # Performing feature engineering
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

    # Removing original columns leaving avg_price and 11 other feature engineered columns
    stock_df = stock_df.copy().drop(['Close','High','Low','Open','Volume','Dividends','Stock Splits'],axis=1)
    stock_df = stock_df.copy().dropna().sort_values('Date')
    cutidx = stock_df.reset_index()[stock_df.index >= '2020-01-01'].index[0] - window
    stock_df = stock_df[stock_df.index >= stock_df.iloc[cutidx].name]

    # Contextually scaling features into window size chunks
    episode_data = []       # for storing scaled data
    stock_data = []         # for storing original data
    idx = 0
    while True:
        fit_data = stock_df.iloc[idx*window:(idx+1)*window]
        scale_data = stock_df.iloc[(idx+1)*window:(idx+2)*window]
        stock_data.extend(scale_data['avg_price'])
        if len(fit_data) and len(scale_data):
            scaler = StandardScaler()
            scaler.fit(fit_data)
            scaled_values = scaler.transform(scale_data)
            episode_data.extend(scaled_values)
            idx += 1
        else:
            break
        
    episode_data, stock_data = np.array(episode_data), np.array(stock_data)
    np.save('episode_2020_scaled.npy', episode_data)
    np.save('episode_2020_prices.npy', stock_data)
    print(episode_data.shape, stock_data.shape)

def make_episode_data(window = 30):
    """Generate contextual scaled episode data for training (2000 to 2019)

    Args:
        window (int, optional): Window size for the episode. Defaults to 30.
    """
    # Downloading data from yahoo finance
    stock_df = yfin.Ticker("VTI").history(start='2000-1-1', end='2019-12-31').sort_values(by='Date')

    # Performing feature engineering
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

    # Contextually scaling features into window size chunks
    episode_data = []       # for storing scaled data
    stock_data = []         # for storing original data
    for i in range(len(stock_df)-2*window):         # Start a new episode for each day
        scaler = StandardScaler()
        scaler.fit(stock_df.iloc[i:i+window])
        scaled_values = scaler.transform(stock_df.iloc[i+window:i+2*window])        # Scaled values for window size with the selected day as first
        episode_data.append(scaled_values)
        stock_data.append(stock_df.iloc[i+window:i+2*window]['avg_price'])
    episode_data, stock_data = np.array(episode_data), np.array(stock_data)
    np.save('episode_till2019_scaled.npy', episode_data)
    np.save('episode_till2019_prices.npy', stock_data)
    print(episode_data.shape, stock_data.shape)

make_episode_data()
make_episode_2020()

class Environment:
    def __init__(self):
        self.episode_data_scaled = np.load('episode_till2019_scaled.npy')     # Scaled episodes of size (episodes, window, state space)
        self.episode_data_prices = np.load('episode_till2019_prices.npy')     # Prices for episodes of size (episodes, window)
        self.window = 30            # Episode window size
        self.episodeidx = 0         # Idx for tracking the episode number
        self.curidx = 0             # Idx for tracking the day within an episode
        self.actions = []           # All actions taken
        self.done = False           # Completion status of episode (ends when curidx reaches window size)

    def seed(self, seed):
        np.random.seed(seed)
    
    def reset(self, idx=None):
        """Reset the environment (start of an episode)

        Args:
            idx (int, optional): Idx of the episode chosen, otherwise randomly selected. Defaults to None.

        Returns:
            np.array: Initial state of the episode
        """
        self.curidx = 0
        self.actions = []
        self.done = False
        self.episodeidx = idx if idx is not None else np.random.randint(len(self.episode_data_scaled))     # Select random episode if not provided
        return self._add_actions_to_state(self.episode_data_scaled[self.episodeidx][0])

    def _add_actions_to_state(self, state):
        """Add 2 extra features to the state as proxy for historical actions

        Args:
            state (np.array): State to which the features need to be added

        Returns:
            np.array: State with progress and buy_ratio as 2 new features
        """
        progress = (self.curidx + 1) / self.window          # Proportion of the episode which has been completed
        buy_ratio = self.actions.count(1) / len(self.actions) if len(self.actions) else 0           # Historically how much purchase actions have been taken
        return np.append(state, [progress, buy_ratio])

    def step(self, action):
        """Takes a step in the environment with the specified action

        Args:
            action (int): Index of the action taken in the current state

        Returns:
            (np.array, float, bool): Returns the next state, reward and completion status of the episode
        """
        self.curidx += 1
        self.actions.append(action)
        self.done = self.curidx == self.window
        
        next_state = None if self.done else self._add_actions_to_state(self.episode_data_scaled[self.episodeidx][self.curidx])
        self.reward = self._reward()
        return next_state, self.reward, self.done
        
    def _reward(self):
        """Calculates the reward

        Returns:
            float: Reward for the current state
        """
        if not self.done:       # If the episode is not over, no reward is calculated
            return 0

        # Reward calculation based on the loss function
        agent_average = np.dot(self.actions, self.episode_data_prices[self.episodeidx]) / self.actions.count(1) if self.actions.count(1) else 0
        daily_average = self.episode_data_prices[self.episodeidx].mean()
        self.first_term = ((agent_average - daily_average) * 2 * self.actions.count(1)) / daily_average
        self.second_term = ((1 - (self.actions.count(1) / (self.window/2))) ** 2)
        reward = -(self.first_term + self.second_term)          # Conversion from minimisation to maximisation problem
        return reward


# Configuration parameters for the whole setup
seed = args.seed
gamma = args.discount_rate  # Discount factor for past rewards
max_steps_per_episode = 31
env = Environment()  # Create the environment
env.seed(seed)
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0

num_inputs = 14
num_actions = 2

inputs = layers.Input(shape=(num_inputs,))
hidden1 = layers.Dense(500, activation="relu")(inputs)
hidden1 = layers.Dropout(0.3)(hidden1)
hidden2 = layers.Dense(250, activation="relu")(hidden1)
hidden2 = layers.Dropout(0.25)(hidden2)
hidden3 = layers.Dense(125, activation="relu")(hidden2)
hidden3 = layers.Dropout(0.25)(hidden3)
hidden4 = layers.Dense(60, activation="relu")(hidden3)
hidden4 = layers.Dropout(0.25)(hidden4)
hidden5 = layers.Dense(30, activation="relu")(hidden4)
hidden5 = layers.Dropout(0.25)(hidden5)
common = layers.Dense(15, activation="relu")(hidden5)
common = layers.Dropout(0.25)(common)
action = layers.Dense(num_actions, activation="softmax")(common)
critic = layers.Dense(1)(common)

model = keras.Model(inputs=inputs, outputs=[action, critic])

initial_learning_rate = args.lr_initial
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=args.lr_decay_steps,
    decay_rate=args.lr_decay_rate,
    staircase=args.lr_staircase
)

optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
huber_loss = keras.losses.Huber()
action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0
max_running_reward = -np.inf

epsilon = args.explore_rate             # Exploration rate
epsilon_min = args.explore_min          # Minimum exploration rate
epsilon_decay = args.explore_decay      # Decay factor for the exploration rate per episode

while True:  # Run until solved
    state = env.reset()
    episode_reward = 0
    log_buy_ratios = []
    log_rewards = []
    with tf.GradientTape() as tape:
        for timestep in range(1, max_steps_per_episode):
            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)

            # Predict action probabilities and estimated future rewards from environment state
            action_probs, critic_value = model(state)
            critic_value_history.append(critic_value[0, 0])

            # Sample action from action probability distribution
            if np.random.random() < epsilon:
                action = np.random.choice(num_actions, p=np.squeeze(action_probs))
            else:
                action = np.argmax(np.squeeze(action_probs))
            action_probs_history.append(tf.math.log(action_probs[0, action]))

            # Apply the sampled action in environment
            state, reward, done = env.step(action)
            rewards_history.append(reward)
            episode_reward += reward

            if done:
                log_buy_ratios.append(env.actions.count(1) / len(env.actions) if len(env.actions) else 0)
                log_rewards.append([reward, env.first_term, env.second_term])
                break

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # Update running reward
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()

        # Calculating loss values to update network
        history = zip(action_probs_history, critic_value_history, returns)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            diff = ret - value
            actor_losses.append(-log_prob * diff)  # actor loss
            critic_losses.append(
                huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            )

        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Clear the loss and reward history
        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()

    # Log details
    episode_count += 1
    if episode_count % 10 == 0:
        template = "running reward: {:.2f} at episode {} with reward {:.2f} and br {:.2f} and rew_1: {:.2f} and rew_2: {:.2f}\n"
        buy_ratio = np.array(log_buy_ratios).mean()
        reward = np.array(log_rewards).mean(axis=0)
        with open('logs', 'a') as f:
            f.write(template.format(running_reward, episode_count, reward[0], buy_ratio, -reward[1], -reward[2]))
        print(template.format(running_reward, episode_count, reward[0], buy_ratio, -reward[1], -reward[2]), end ='')
        log_buy_ratios.clear()
        log_rewards.clear()

    if running_reward > max_running_reward:  # Condition to consider the task solved
        max_running_reward = running_reward
        model.save('model_best')
    
    if episode_count % args.model_save_freq == 0:
        model.save(f'model_{episode_count}')
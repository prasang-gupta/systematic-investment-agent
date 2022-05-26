import yfinance as yfin
import pandas as pd
import random
import numpy as np
from tqdm import tqdm
from collections import deque
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser(description='Train DQN agent')
parser.add_argument('--total_episodes', action='store', type=int, default=1440,
                    help='number of total episodes (default: 1440)')
parser.add_argument('--discount_rate', action='store', type=float, default=0.95,
                    help='discount rate for past rewards (default: 0.95)')
parser.add_argument('--episodes_target_copy', action='store', type=int, default=2,
                    help='number of episodes after which the model weights are copied to the target model (default: 2)')
parser.add_argument('--episodes_test', action='store', type=int, default=5,
                    help='number of episodes after which the test environment is run to evaluate agent on 2020 data (default: 5)')
parser.add_argument('--er_batch_size', action='store', type=int, default=32,
                    help='experience replay batch size (default: 32)')
parser.add_argument('--er_buffer_size', action='store', type=int, default=7500,
                    help='experience replay total buffer size (default: 7500)')
parser.add_argument('--unusual_sample_factor', action='store', type=float, default=0.9,
                    help='unusual sampling factor for experience replay (default: 0.9)')
parser.add_argument('--explore_rate', action='store', type=float, default=1.0,
                    help='epsilon (default: 1.0)')
parser.add_argument('--explore_min', action='store', type=float, default=0.01,
                    help='minimum epsilon for exploration (default: 0.01)')
parser.add_argument('--explore_decay', action='store', type=float, default=0.999,
                    help='decay factor for exploration rate (default: 0.999)')
args = parser.parse_args()

## CONSTANTS
EPISODE_TIME = 30                   # Time for each episode (window size)
STATE_SIZE = 12+2                   # State space size
ACTION_SIZE = 2                     # Action space size

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

class ExperienceReplay:
    def __init__(self, buffer_size, unusual_sample_factor):
        """Store and sample samples from the environment experience with unusual sampling

        Args:
            buffer_size (int): Max size of the buffer to be maintained
            unusual_sample_factor (float): Factor to give more weightage to non-zero reward samples. 0 = Extreme weightage, 1 = Uniform sampling
        """
        self.buffer = []
        self.buffer_size = buffer_size
        self.unusual_sample_factor = unusual_sample_factor
    
    def add(self, experience):
        """Add an experience to the buffer

        Args:
            experience (tuple): state, action, reward, next_state, done
        """
        self.buffer.append(experience)
        self.buffer = self.buffer[-self.buffer_size:]
        
    def sample(self, size):
        """Sample a batch of experiences from the buffer

        Args:
            size (int): Size of the sample requested

        Returns:
            np.array: Sampled experiences 
        """
        buffer = sorted(self.buffer, key=lambda replay: abs(replay[2]), reverse=True)
        p = np.array([self.unusual_sample_factor ** i for i in range(len(buffer))])
        p = p / sum(p)
        sample_idxs = np.random.choice(np.arange(len(buffer)), size=size, p=p)
        sample_output = [buffer[idx] for idx in sample_idxs]
        sample_output = np.reshape(sample_output, (size,-1))
        return sample_output

class DQNAgent:
    def __init__(self, state_size, action_size):
        """Deep Q Network Agent

        Args:
            state_size (int): Size of the state space
            action_size (int): Size of the action space
        """
        self.state_size = state_size                    # State space size
        self.action_size = action_size                  # Action space size
        self.memory = ExperienceReplay(buffer_size=args.er_buffer_size, unusual_sample_factor=args.unusual_sample_factor)         # Memory for storing and sampling experiences
        self.gamma = args.discount_rate               # Discount rate
        self.epsilon = args.explore_rate              # Exploration rate
        self.epsilon_min = args.explore_min         # Minimum exploration rate
        self.epsilon_decay = args.explore_decay      # Decay factor for the exploration rate per episode
        self.model = self._build_model()                # NN model for the agent
        self.target_model = self._build_model()         # NN model to be used as the target model for the agent

    def _build_model(self):
        """Builds the model architecture

        Returns:
            keras_model: DQN Model
        """

        initial_learning_rate = 0.001
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.99,
            staircase=True
        )
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

        model = keras.models.Sequential()
        model.add(keras.layers.Dense(500, input_dim=self.state_size, activation='relu'))
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Dense(250, activation='relu'))
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Dense(125, activation='relu'))
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Dense(60, activation='relu'))
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Dense(30, activation='relu'))
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Dense(15, activation='relu'))
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=optimizer)
        return model

    def memorize(self, state, action, reward, next_state, done):
        """Store experience in memory

        Args:
            state (np.array): Current environment state
            action (int): Current action taken
            reward (float): Current reward
            next_state (np.array): Next environment state
            done (bool): True if episode over, otherwise False
        """
        self.memory.add((state, action, reward, next_state, done))

    def act(self, state):
        """Get action for the current state (Epsilon-greedy)

        Args:
            state (np.array): Current state of the environment

        Returns:
            int: Index of the action taken by the agent in the current state
        """
        if np.random.rand() <= self.epsilon:            # Chose a random action with probability epsilon
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])                 # Chose the best action (with the highest Q value)

    def replay(self, batch_size):
        """Replay experiences and train the model

        Args:
            batch_size (int): Batch size for the experience replay
        """
        minibatch = self.memory.sample(batch_size)              # Sample a batch of samples
        for state, action, reward, next_state, done in minibatch:
            target = reward if done else reward + (self.gamma * np.amax(self.target_model.predict(next_state)[0]))      # Get the target Q value
            target_f = self.model.predict(state)                # Get the current state value
            target_f[0][action] = target                        # Update the target Q value in the current value
            self.model.fit(state, target_f, epochs=1, verbose=0)            # Train the model for 1 epoch

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay              # Decay the epsilon

    def load(self, model, name):
        """Load model weights

        Args:
            model (keras_model): Model architecture
            name (str): Name of the file
        """
        model.load_weights(name)

    def save(self, model, name):
        """Save model weights

        Args:
            model (keras_model): Model architecture
            name (str): Name of the file
        """
        model.save_weights(name)

    def copy_weights_to_target(self):
        """Copy weights from the DQN model to the target model
        """
        self.save(self.model, 'model_weights.h5')
        self.load(self.target_model, 'model_weights.h5')

class TestEnvironment:
    def __init__(self, window=30):
        """Test Environment to run the agent on the 2020 dataset (out of train)

        Args:
            window (int, optional): Window size for the episode. Defaults to 30.
        """
        self.data = np.load('episode_2020_scaled.npy')                 # Scaled data
        self.stock_data = np.load('episode_2020_prices.npy')     # Original data
        self.window = window            # Window size

    def score(self, model):
        """Score the agent performance

        Args:
            model (keras_model): DQN model
        """
        months = len(self.data) // self.window          # Distribute the data in chunks of window size
        with open('tests', 'a') as f:
            f.write('--------------------------------------------\n')
            for month in range(months):
                daily_price, agent_price, buy_ratio = self._score_month(month, model)       # Write prices and buy_ratio for every month
                f.write(f'{daily_price}\t{agent_price}\t{buy_ratio}\n')

    def _score_month(self, month, model):
        """Calculate agent score for a month

        Args:
            month (int): Idx of the month for which score needs to be calculated
            model (keras_model): DQN model

        Returns:
            (float, float, float): Daily average price, Agent average price and buy ratio for the agent
        """
        data = self.data[month*self.window:(month+1)*self.window]               # Scaled data for the month
        price = self.stock_data[month*self.window:(month+1)*self.window]        # Original prices for the month
        actions = []
        idx = 0
        for state in data:                                  # Run the model for the whole month, one day at a time storing actions
            progress = (idx + 1) / self.window
            buy_ratio = actions.count(1) / len(actions) if len(actions) else 0
            state = np.append(state, [progress, buy_ratio]).reshape(1,-1)
            action = np.argmax(model.predict(state)[0])
            actions.append(action)
            idx += 1
        daily_price = price.mean()              # Average daily price
        agent_price = 0 if not actions.count(1) else np.dot(actions, price) / actions.count(1)          # Average agent price
        return daily_price, agent_price, actions.count(1) / len(actions)

agent = DQNAgent(STATE_SIZE, ACTION_SIZE)
env = Environment()
testenv = TestEnvironment(EPISODE_TIME)

for e in range(args.total_episodes):           # For each episode
    state = env.reset(e)            # Get the starting state for 'e' episode
    state = state.reshape(1, -1)
    
    if e % args.episodes_target_copy == 0:           # Copy model weights to target model
        agent.copy_weights_to_target()
    
    for time in tqdm(range(EPISODE_TIME)):            # For each day in the episode
        action = agent.act(state)               # Get the action of the agent (epsilon-greedy)
        next_state, reward, done = env.step(action)     # Step the environment based on current action taken
        if not done:
            next_state = next_state.reshape(1, -1)
        agent.memorize(state, action, reward, next_state, done)         # Store the instance in experience replay
        state = next_state              # Set the current state as the new state
        if done:                # If episode is over
            buy_ratio = env.actions.count(1) / len(env.actions) if len(env.actions) else 0
            with open('logs', 'a') as f:
                f.write(f"episode: {e}/{args.total_episodes}, reward: {env.reward}, rew_1: {-env.first_term}, rew_2: {-env.second_term}, buy_ratio: {buy_ratio}\n")
            break
        if len(agent.memory.buffer) > args.er_batch_size:           # If there are atleast batch size number of samples in experience replay buffer
            agent.replay(args.er_batch_size)                        # Train the model for a batch of experiences

    agent.decay_epsilon()
    
    if e % args.episodes_test == 0:              # Run the agent on the test environment
        testenv.score(agent.model)




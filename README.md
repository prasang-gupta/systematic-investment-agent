# systematic-investment-agent

![python](https://img.shields.io/badge/python-v3.8.12-green)
![tensorflow](https://img.shields.io/badge/tensorflow-v2.8.0-blue)

This code is for replicating experiments presented in the paper 'Intelligent Systematic Investment Agent: an ensemble of deep learning and evolutionary strategies' and submmited to NeurIPS 2022. This page contains step by step instructions on replicating the experiments presented in the paper and instructions on creating new ones to stress test the model. The code is written purely in python and all dependencies are included in requirements.txt.

#### Repository structure

    .
    ├── actorcritic
    │   ├── pretrained          # Pre-trained weights and logs (submitted)
    │   │   └── ...
    │   ├── train.py            # Train actor critic agent
    │   └── val.py              # Get results from trained actor critic agent
    ├── dqn
    │   ├── pretrained          # Pre-trained weights and logs (submitted)
    │   │   └── ...
    │   ├── train.py            # Train DQN agent
    │   └── val.py              # Get results from trained DQN agent
    ├── ga
    │   ├── pretrained          # Pre-trained weights and logs (submitted)
    │   │   └── ...
    │   └── scripts
    │       ├── dataprep.py     # Run GA and get optimum action vectors
    │       ├── train.py        # Train NN on GA output
    │       ├── val.py          # Get results from trained GADLE algorithm
    │       └── ...
    ├── rawdata.pkl             # Stored raw ETF data
    ├── requirements.txt        # Requirements file for environment setup
    └── ...

#### Dependencies

The code is dependent on packages that can be installed by running the requirements file.
To install requirements:

```
pip install -r requirements.txt
```

#### GADLE algorithm

##### For reproducing results

- Change to the `ga` directory and run the `val.py` script as:

```bash
cd ga
python scripts/val.py
```

##### For training fresh model

- `scripts/dataprep.py` : This generates episodic samples, does feature engineering, scaling and finds the optimal action vectors for all episodes using genetic algorithm that is subsequently used as a training dataset for the nueral network.
```
usage: dataprep.py [-h] [--etf ETF] [--end_date END_DATE] [--episodes EPISODES]

Generate episodic samples, perform feature engineering, scaling and find the optimal action vectors using GA

optional arguments:
  -h, --help           show this help message and exit
  --etf ETF            index fund identifier (default: VTI)
  --end_date END_DATE  period upto which training is considered (default: 2019-12-31)
  --episodes EPISODES  number of episodes to generate (default: 4245)
```

- `scripts/train.py` : This trains the agent to learn the policy behind optimal action using a simple neural network and saves the model to disk.

```
usage: train.py [-h] [--etf ETF] [--end_date END_DATE] [--epochs EPOCHS]

Learn the policy behind optimal action using Neural Network architecture

optional arguments:
  -h, --help           show this help message and exit
  --etf ETF            index fund identifier (default: VTI)
  --end_date END_DATE  period upto which training is considered (default: 2019-12-31)
  --epochs EPOCHS      number of epochs for training (default: 150)
```

- `scripts/val.py` : This runs the saved model on the unseen 2020 dataset and validates the performance.

```
usage: val.py [-h] [--path PATH] [--start_date START_DATE]

Get the results for the trained GADLE model

optional arguments:
  -h, --help            show this help message and exit
  --path PATH           path where model outputs are stored (default: pretrained)
  --start_date START_DATE
                        start date for validation period (default: 2020-1-1)
```

#### Actor Critic agent

##### For reproducing results

- Change to the `actorcritic` directory and run the `val.py` script as:

```bash
cd actorcritic
python val.py
```

##### For training fresh model

- `train.py` : This trains the new actor critic agent from scratch (with controlled exploration and LR-scheduled NN training with the same number of parameters as the GADLE algorithm) and saves the model files and logs in the same folder.

```
usage: train.py [-h] [--discount_rate DISCOUNT_RATE] [--lr_initial LR_INITIAL] [--lr_decay_steps LR_DECAY_STEPS] [--lr_decay_rate LR_DECAY_RATE]
                [--lr_staircase LR_STAIRCASE] [--explore_rate EXPLORE_RATE] [--explore_min EXPLORE_MIN] [--explore_decay EXPLORE_DECAY]
                [--model_save_freq MODEL_SAVE_FREQ] [--seed SEED]

Train Actor Critic agent

optional arguments:
  -h, --help            show this help message and exit
  --discount_rate DISCOUNT_RATE
                        discount rate for past rewards (default: 0.97)
  --lr_initial LR_INITIAL
                        initial learning rate (default: 0.001)
  --lr_decay_steps LR_DECAY_STEPS
                        learning rate decay steps (default: 1000)
  --lr_decay_rate LR_DECAY_RATE
                        learning rate decay rate (default: 0.99)
  --lr_staircase LR_STAIRCASE
                        staircase for learning rate decay (default: True)
  --explore_rate EXPLORE_RATE
                        epsilon (default: 1.0)
  --explore_min EXPLORE_MIN
                        minimum epsilon for exploration (default: 0.01)
  --explore_decay EXPLORE_DECAY
                        decay factor for exploration rate (default: 0.999)
  --model_save_freq MODEL_SAVE_FREQ
                        save model after these many episodes (default: 10000)
  --seed SEED           seed for training (default: 42)
```

- `val.py` : This runs the saved model on the unseen 2020 dataset and validates the performance.

```
usage: val.py [-h] [--path PATH] [--model MODEL]

Get results for Actor Critic agent training

optional arguments:
  -h, --help     show this help message and exit
  --path PATH    path where model outputs are stored (default: pretrained)
  --model MODEL  model file to get results for (default: latest file)
```

#### DQN agent

##### For reproducing results

- Change to the `dqn` directory and run the `val.py` script as:

```bash
cd dqn
python val.py
```

##### For training fresh model

- `train.py` : This trains the new DQN agent from scratch (with controlled exploration, unusual sampled experience replay and LR-scheduled NN training with the same number of parameters as the GADLE algorithm) and saves the model files and logs in the same folder.

```
usage: train.py [-h] [--total_episodes TOTAL_EPISODES] [--discount_rate DISCOUNT_RATE] [--episodes_target_copy EPISODES_TARGET_COPY]
                [--episodes_test EPISODES_TEST] [--er_batch_size ER_BATCH_SIZE] [--er_buffer_size ER_BUFFER_SIZE]
                [--unusual_sample_factor UNUSUAL_SAMPLE_FACTOR] [--explore_rate EXPLORE_RATE] [--explore_min EXPLORE_MIN]
                [--explore_decay EXPLORE_DECAY]

Train DQN agent

optional arguments:
  -h, --help            show this help message and exit
  --total_episodes TOTAL_EPISODES
                        number of total episodes (default: 1440)
  --discount_rate DISCOUNT_RATE
                        discount rate for past rewards (default: 0.95)
  --episodes_target_copy EPISODES_TARGET_COPY
                        number of episodes after which the model weights are copied to the target model (default: 2)
  --episodes_test EPISODES_TEST
                        number of episodes after which the test environment is run to evaluate agent on 2020 data (default: 5)
  --er_batch_size ER_BATCH_SIZE
                        experience replay batch size (default: 32)
  --er_buffer_size ER_BUFFER_SIZE
                        experience replay total buffer size (default: 7500)
  --unusual_sample_factor UNUSUAL_SAMPLE_FACTOR
                        unusual sampling factor for experience replay (default: 0.9)
  --explore_rate EXPLORE_RATE
                        epsilon (default: 1.0)
  --explore_min EXPLORE_MIN
                        minimum epsilon for exploration (default: 0.01)
  --explore_decay EXPLORE_DECAY
                        decay factor for exploration rate (default: 0.999)
```

- `val.py` : This runs the saved model on the unseen 2020 dataset and validates the performance.

```
usage: val.py [-h] [--path PATH]

Get results for DQN agent training

optional arguments:
  -h, --help   show this help message and exit
  --path PATH  path where model outputs are stored (default: pretrained)
```
import numpy as np
import gym

ENV_NAME = 'Taxi-v3'

env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
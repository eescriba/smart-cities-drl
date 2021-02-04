from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Embedding, Reshape
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

from ..env import env

action_size = env.action_space.n


def build_model():
    model = Sequential()
    model.add(Embedding(500, 10, input_length=1))
    model.add(Reshape((10,)))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(action_size, activation="linear"))
    print(model.summary())
    return model


model = build_model()
memory = SequentialMemory(limit=50000, window_length=1)
policy = EpsGreedyQPolicy()
dqn = DQNAgent(
    model=model,
    nb_actions=action_size,
    memory=memory,
    nb_steps_warmup=500,
    target_model_update=1e-2,
    policy=policy,
)
dqn.compile(Adam(lr=1e-3), metrics=["mae"])
dqn.load_weights("weights/dqn_Taxi-v3_weights.h5f")

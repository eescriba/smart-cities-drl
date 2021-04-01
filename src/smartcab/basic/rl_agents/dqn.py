from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Embedding, Reshape
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

# action_size = env.action_space.n
action_size = 6
state_size = 720


def build_model():
    model = Sequential()
    model.add(Embedding(state_size, 10, input_length=1))
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
# dqn.fit(
#     env,
#     nb_steps=100000,
#     visualize=False,
#     verbose=1,
#     nb_max_episode_steps=999,
#     log_interval=10000,
# )
# dqn.save_weights("weights/dqn_city5_weights.h5f", overwrite=True)
# dqn.load_weights("weights/dqn_basic_weights.h5f")

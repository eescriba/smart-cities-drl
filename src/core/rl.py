""" RL """
from abc import ABC, ABC

import ray
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune import run, sample_from
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.tune.schedulers import PopulationBasedTraining
import tensorflow as tf


RAY_DIR = "ray_results/"


class RLlibAgent(ABC):
    def __init__(self, name, config, env_class, env_config):
        ray.init(ignore_reinit_error=True)
        self.name = name
        self.config = config
        self.env_class = env_class
        self.env_config = env_config
        self.config.setdefault("num_workers", 1)
        self.config.setdefault("num_gpus", len(tf.config.list_physical_devices("GPU")))
        self.agent = self.agent_class(config=config, env=env_class)

    @property
    def agent_class(self):
        raise NotImplementedError("Subclasses should implement agent_class field")

    def load(self, path):
        """
        Load a trained RLlib agent from the specified checkpoint path.
        """
        self.agent.restore(path)

    def tune(self, config, stop_criteria, scheduler=None):
        """
        Tune hyperparameters for a RLlib agent
        """
        analysis = run(
            self.agent.__class__,
            name=self.name,
            config=config,
            local_dir=RAY_DIR,
            stop=stop_criteria,
            scheduler=scheduler,
            checkpoint_at_end=True,
        )
        return analysis

    def train(self, stop_criteria):
        """
        Train a RLlib agent
        """
        analysis = self.tune(self.config, stop_criteria)
        checkpoint_path = analysis.get_trial_checkpoints_paths(
            trial=analysis.get_best_trial("episode_reward_mean"),
            metric="episode_reward_mean",
        )[0][0]
        return analysis, checkpoint_path

    def test(self, num_steps):
        """
        Test trained agent for specified number of episodes
        """
        env = self.env_class(self.env_config)
        obs = env.reset()
        done = False
        episode_reward = 0
        sum_reward = 0

        for step in range(n_step):
            action = self.agent.compute_action(obs)
            print(action)
            state, reward, done, info = env.step(action)
            print(state, reward, done, info)
            sum_reward += reward
            if done:
                print("cumulative reward", sum_reward)
                state = env.reset()
                sum_reward = 0

        return sum_reward


class PPOAgent(RLlibAgent):
    agent_class = PPOTrainer


class DQNAgent(RLlibAgent):
    agent_class = DQNTrainer

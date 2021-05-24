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

    @property
    def algorithm(self):
        raise NotImplementedError("Subclasses should implement algorithm field")

    def load(self, path):
        """
        Load a trained RLlib agent from the specified checkpoint path.
        """
        self.agent.restore(path)

    def tune(self, config, stop_criteria, num_samples=8, scheduler=None):
        """
        Tune hyperparameters for a RLlib agent
        """
        analysis = run(
            self.agent_class,
            name=self.name,
            config=config,
            local_dir=RAY_DIR,
            stop=stop_criteria,
            scheduler=scheduler,
            num_samples=num_samples,
            checkpoint_at_end=True,
        )
        return analysis

    def train(self, stop_criteria):
        """
        Train a RLlib agent
        """
        analysis = self.tune(self.config, stop_criteria, num_samples=1)
        checkpoint_path = analysis.get_trial_checkpoints_paths(
            trial=analysis.get_best_trial("episode_reward_mean"),
            metric="episode_reward_mean",
        )[0][0]
        return analysis, checkpoint_path

    def test(self, num_episodes, verbose=False):
        """
        Test trained agent for specified number of episodes
        """
        env = self.env_class(self.env_config)
        mean_reward = 0
        max_reward = 0
        min_reward = 0
        for n in range(num_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = self.agent.compute_action(obs)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
            if verbose:
                print("Episode reward: ", episode_reward)
            mean_reward += episode_reward
            max_reward = max(max_reward, episode_reward)
            min_reward = min(min_reward, episode_reward)
        mean_reward /= num_episodes
        if verbose:
            print("-----------------------")
            print("Min reward: ", min_reward)
            print("Max reward: ", max_reward)
            print("Avg reward: ", episode_reward)
        return mean_reward, min_reward, max_reward


class PPOAgent(RLlibAgent):
    agent_class = PPOTrainer


class DQNAgent(RLlibAgent):
    agent_class = DQNTrainer

from ray.tune.schedulers import PopulationBasedTraining


class PbtOptimizer:
    def __init__(self, hyperparam_mutations):
        self.scheduler = PopulationBasedTraining(
            time_attr="time_total_s",
            perturbation_interval=120,
            resample_probability=0.25,
            metric="episode_reward_mean",
            mode="max",
            hyperparam_mutations=hyperparam_mutations,
            custom_explore_fn=self.explore,
        )

    # Postprocess the perturbed config to ensure it's still valid used if PBT.
    def explore(config):
        # ensure we collect enough timesteps to do sgd
        if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
            config["train_batch_size"] = config["sgd_minibatch_size"] * 2
        # ensure we run at least one sgd iter
        if config["num_sgd_iter"] < 1:
            config["num_sgd_iter"] = 1
        return config

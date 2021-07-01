from ray.tune.schedulers import PopulationBasedTraining


class PbtOptimizer:
    def __init__(
        self,
        hyperparam_mutations,
        time_attr="time_total_s",
        perturbation_interval=60,
        metric="episode_reward_mean",
        mode="max",
    ):
        self.scheduler = PopulationBasedTraining(
            time_attr=time_attr,
            perturbation_interval=perturbation_interval,
            resample_probability=0.25,
            metric=metric,
            mode=mode,
            hyperparam_mutations=hyperparam_mutations,
            custom_explore_fn=self.explore,
        )

    @staticmethod
    def explore(config):
        # Postprocess the perturbed config to ensure it's still valid used if PBT
        # ensure we collect enough timesteps to do sgd
        if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
            config["train_batch_size"] = config["sgd_minibatch_size"] * 2
        # ensure we run at least one sgd iter
        if config["num_sgd_iter"] < 1:
            config["num_sgd_iter"] = 1
        return config

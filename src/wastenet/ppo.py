from ray.rllib.agents.ppo import DEFAULT_CONFIG

default_config = DEFAULT_CONFIG.copy()

best_config = {
    "observation_filter": "MeanStdFilter",
    "model": {"free_log_std": True},
    "num_sgd_iter": 10,
    "sgd_minibatch_size": 128,
    "lambda": 0.731396,
    "clip_param": 0.317651,
    "lr": 5e-05,
    "train_batch_size": 18812,
}

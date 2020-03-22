from spinup_vpg import vpg
from ambiguous_bandit import AmbiguousBanditEnv

import wandb


config = dict(
    ac_kwargs={'hidden_sizes': [256, 256]},
    gamma=0.99,
    seed=0,
    max_ep_len=1,
    steps_per_epoch=2,
    epochs=100,
)

if __name__ == '__main__':
    wandb.init(project="hca", config=config, tags=['ambiguous_bandit', 'vpg'])
    logger_kwargs={'exp_name': 'hca', 'output_dir': wandb.run.dir}
    vpg(env_fn=AmbiguousBanditEnv, **config, logger_kwargs=logger_kwargs)

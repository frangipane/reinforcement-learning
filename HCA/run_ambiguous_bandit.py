import time
import wandb

from spinup_vpg import vpg
from ambiguous_bandit import AmbiguousBanditEnv
from utils import plot_test_returns


config = dict(
    ac_kwargs={'hidden_sizes': []},
    gamma=0.99,
    seed=0,
    max_ep_len=1,
    steps_per_epoch=128,
    epochs=100,
    n_test_episodes=100
)

if __name__ == '__main__':
    wandb.init(project="hca", config=config, tags=['ambiguous_bandit', 'vpg'])
    logger_out_dir = wandb.run.dir
    #logger_out_dir = f'/tmp/hca/{int(time.time())}'
    logger_kwargs={'exp_name': 'hca', 'output_dir': logger_out_dir}
    vpg(env_fn=AmbiguousBanditEnv, **config, logger_kwargs=logger_kwargs)

    plot_test_returns(logger_out_dir, 'progress.txt')

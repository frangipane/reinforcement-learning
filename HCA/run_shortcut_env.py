import os
import pandas as pd
import matplotlib.pyplot as plt

from spinup_vpg import vpg
from shortcut_env import ShortcutEnv

import time
import wandb


def plot_test_returns(returns_path_dir, returns_path_fname):
    df = pd.read_table(os.path.join(returns_path_dir, returns_path_fname))
    x = df.Epoch
    y = df.AverageTestEpRet
    std = df.StdTestEpRet

    fig, ax = plt.subplots(1,1)
    ax.plot(x, y)
    ax.fill_between(x, y-std, y+std, alpha=0.5)
    ax.set_xlabel('epoch')
    ax.set_ylabel('average return')
    ax.set_title('Average returns per epoch for test agent')
    plt.show()
    fig.savefig(os.path.join(returns_path_dir, 'avg_test_returns'), fmt='png')


config = dict(
    ac_kwargs={'hidden_sizes': [64]},
    pi_lr=0.001,
    gamma=0.99,
    seed=0,
    max_ep_len=10000, # chain length of 5, so this will never be hit
    steps_per_epoch=5,
    epochs=100,
    n_test_episodes=100
)

if __name__ == '__main__':
    wandb.init(project="hca", config=config, tags=['shortcut_env', 'vpg'])
    logger_out_dir = wandb.run.dir
    logger_kwargs={'exp_name': 'hca', 'output_dir': logger_out_dir}
    vpg(env_fn=ShortcutEnv, **config, logger_kwargs=logger_kwargs)

    plot_test_returns(logger_out_dir, 'progress.txt')

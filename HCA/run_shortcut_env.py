import time
import wandb

#from spinup_vpg import vpg
from tabular_vpg import vpg
import hca.envs.shortcut_env
import tabular_actor_critic
from utils import plot_test_returns

# spinup_vpg config
# config = dict(
#     ac_kwargs={'hidden_sizes': [64]},
#     pi_lr=0.001,
#     gamma=0.99,
#     seed=0,
#     max_ep_len=10000, # chain length of 5, so this will never be hit
#     steps_per_epoch=5,
#     epochs=100,
#     n_test_episodes=100
# )

# tabular_vpg config
config = dict(
    env_kwargs={'OHE_obs': False, 'random_start': True, 'n': 5},
    ac_kwargs={'pi_lr': 0.4, 'vf_lr': 0.4},
    n_episodes=100,
    n_test_episodes=100,
    gamma=0.99,
    lam=0.95,
)

if __name__ == '__main__':
    wandb.init(project="hca", config=config, tags=['shortcut_env', 'tabular_vpg'])
    logger_out_dir = wandb.run.dir
    logger_kwargs={'exp_name': 'hca', 'output_dir': logger_out_dir}
    vpg(shortcut_env.ShortcutEnv, **config, actor_critic=tabular_actor_critic.TabularVPGActorCritic,
        logger_kwargs=logger_kwargs)

    plot_test_returns(logger_out_dir, 'progress.txt')

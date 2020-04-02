import time
import numpy as np
import wandb

#from hca.spinup_vpg import vpg
import hca.tabular_actor_critic as tabular_actor_critic
from hca.tabular_vpg import vpg
from hca.envs.ambiguous_bandit import AmbiguousBanditEnv
from hca.utils import plot_test_returns

# spinup_vpg config
# config = dict(
#     ac_kwargs={'hidden_sizes': []},
#     gamma=0.99,
#     seed=0,
#     pi_lr=0.2,
#     vf_lr=0.2,
#     max_ep_len=1,
#     steps_per_epoch=2,
#     epochs=100,
#     n_test_episodes=100
# )

# tabular_vpg config
# config = dict(
#     env_kwargs={'OHE_obs': False,
#                 'means': {'HI': 2, 'LO': 1},
#                 'stds': {'HI': 1.5, 'LO': 1.5},
#                 'epsilon': 0.1},
#     ac_kwargs={'pi_lr': 0.2, 'vf_lr': 0.2},
#     n_episodes=100,
#     n_test_episodes=100,
#     gamma=1.0,
#     lam=1.0,
#     actor_critic=tabular_actor_critic.TabularVPGActorCritic,
#     algo='vpg',
# )

# return HCA config
# config = dict(
#     env_kwargs={'OHE_obs': False,
#                 'means': {'HI': 2, 'LO': 1},
#                 'stds': {'HI': 1.5, 'LO': 1.5},
#                 'epsilon': 0.1},
#     ac_kwargs={'pi_lr': 0.3, 'vf_lr': 0.3, 'h_lr': 0.3,
#                'return_bins': np.arange(-.975,4.5,0.55)},
#     n_episodes=100,
#     n_test_episodes=100,
#     gamma=1.0,
#     lam=1.0,
#     actor_critic=tabular_actor_critic.TabularReturnHCA,
#     algo='returnHCA',
# )

# state HCA config
config = dict(
    env_kwargs={'OHE_obs': False,
                'means': {'HI': 2, 'LO': 1},
                'stds': {'HI': 1.5, 'LO': 1.5},
                'epsilon': 0.1},
    ac_kwargs={'pi_lr': 0.3, 'vf_lr': 0.3, 'h_lr': 0.4},
    n_episodes=100,
    n_test_episodes=100,
    gamma=1.0,
    lam=1.0,
    actor_critic=tabular_actor_critic.TabularStateHCA,
    algo='stateHCA',

)


if __name__ == '__main__':
    algo = config.pop('algo')

    wandb.init(
        project="hca",
        config=config,
        name=algo + f"|AmbiguousBanditEnv",
        tags=['ambiguous_bandit', algo])
    logger_out_dir = wandb.run.dir
    logger_kwargs={'exp_name': 'hca', 'output_dir': logger_out_dir}
    vpg(env_fn=AmbiguousBanditEnv, **config, logger_kwargs=logger_kwargs)

    plot_test_returns(logger_out_dir, 'progress.txt')

import time
import numpy as np
import wandb

import hca.tabular_actor_critic as tabular_actor_critic
from hca.tabular_vpg import vpg
from hca.envs.delayed_effect_env import DelayedEffectEnv
from hca.utils import plot_test_returns


# # tabular_vpg config
# config = dict(
#     env_kwargs={'OHE_obs': False,
#                 'n': 4,
#                 'final_reward': 1.0,
#                 'sigma': 0.0},
#     ac_kwargs={'pi_lr': 0.2, 'vf_lr': 0.2},
#     n_episodes=500,
#     n_test_episodes=100,
#     gamma=1.0,
#     lam=1.0,
#     actor_critic=tabular_actor_critic.TabularVPGActorCritic,
#     algo='vpg',
#     bootstrap_n=3
# )

# # return HCA config
# config = dict(
#     env_kwargs={'OHE_obs': False,
#                 'n': 4,
#                 'final_reward': 1.0,
#                 'sigma': 0.0},
#     ac_kwargs={'pi_lr': 0.2, 'vf_lr': 0.2, 'h_lr': 0.2, 'return_bins': np.array([-1,0,1])},
#     n_episodes=500,
#     n_test_episodes=100,
#     gamma=1.0,
#     lam=1.0,
#     actor_critic=tabular_actor_critic.TabularReturnHCA,
#     algo='returnHCA',
#     bootstrap_n=3
# )

# state HCA config
config = dict(
    env_kwargs={'OHE_obs': False,
                'n': 4,
                'final_reward': 1.0,
                'sigma': 0.0},
    ac_kwargs={'pi_lr': 0.2, 'vf_lr': 0.2, 'h_lr': 0.4},
    n_episodes=500,
    n_test_episodes=100,
    gamma=1.0,
    lam=1.0,
    actor_critic=tabular_actor_critic.TabularStateHCA,
    algo='stateHCA',
    bootstrap_n=3
)


if __name__ == '__main__':
    algo = config.pop('algo')
    bootstrap = str(config['bootstrap_n'])

    wandb.init(
        project="hca",
        config=config,
        name=algo + f"|DelayedEffectEnv|sigma={config['env_kwargs']['sigma']}|{bootstrap}-step",
        tags=['delayed_effect', algo, bootstrap])
    logger_out_dir = wandb.run.dir
    logger_kwargs={'exp_name': 'hca', 'output_dir': logger_out_dir}

    vpg(env_fn=DelayedEffectEnv, **config, logger_kwargs=logger_kwargs)

    plot_test_returns(logger_out_dir, 'progress.txt')

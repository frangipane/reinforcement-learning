import time
import wandb

#from spinup_vpg import vpg
import tabular_actor_critic
from tabular_vpg import vpg
from ambiguous_bandit import AmbiguousBanditEnv
from utils import plot_test_returns

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
config = dict(
    env_kwargs={'OHE_obs': False,
                'means': {'HI': 2, 'LO': 1},
                'stds': {'HI': 1.5, 'LO': 1.5},
                'epsilon': 0.1},
    ac_kwargs={'pi_lr': 0.2, 'vf_lr': 0.2},
    n_episodes=100,
    n_test_episodes=100,
    gamma=0.99,
    lam=0.95,
)

if __name__ == '__main__':
    wandb.init(project="hca", config=config, tags=['ambiguous_bandit', 'tabular_vpg'])
    logger_out_dir = wandb.run.dir
    logger_kwargs={'exp_name': 'hca', 'output_dir': logger_out_dir}
    vpg(env_fn=AmbiguousBanditEnv, **config, actor_critic=tabular_actor_critic.TabularVPGActorCritic,
        logger_kwargs=logger_kwargs)

    plot_test_returns(logger_out_dir, 'progress.txt')

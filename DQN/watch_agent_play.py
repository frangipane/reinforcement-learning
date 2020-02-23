"""
Helper script to load a trained model and watch it play for a certain number of episodes, with
option to output recorded video of game play.
"""
import time
import torch
import wandb
import numpy as np
import warnings

from gym.wrappers import Monitor
from spinup.utils.logx import Logger

from atari_wrappers import make_atari, wrap_deepmind
from nnetworks import MLPCritic, CNNCritic


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def play_game(env,
              torch_load_kwargs={},
              actor_critic=CNNCritic,
              episodes=10,
              render=False,
              logger_kwargs={}):

    logger = Logger(**logger_kwargs)
    logger.save_config(locals())

    ac = actor_critic(env.observation_space, env.action_space)

    # model saved on GPU, load on CPU: https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-across-devices
    ac_saved = torch.load(**torch_load_kwargs)
    ac_saved = ac_saved.to(device)
    ac.q.load_state_dict(ac_saved.q.module.state_dict())
    ac.q.to(device)

    avg_ret = 0
    avg_raw_ret = 0
    game = 0

    for ep in range(episodes):
        o, ep_ret, ep_len, d, raw_ret  = env.reset(), 0, 0, False, 0
        while not d:
            if render:
                env.render()
            o = torch.as_tensor(o, dtype=torch.float32, device=device)
            o2, r, d, info = env.step(ac.act(o))
            ep_ret += r
            ep_len += 1
            o = o2

        print(f'Returns for episode {ep}: {ep_ret}')
        avg_ret += (1./(ep+1)) * (ep_ret - avg_ret)

        lives = info.get('ale.lives')
        if lives is not None and lives == 0:
            raw_rew = env.get_episode_rewards()[-1]
            raw_len = env.get_episode_lengths()[-1]
            logger.log_tabular('RawRet', raw_rew)
            logger.log_tabular('RawLen', raw_len)
            logger.log_tabular('GameId', game)
            wandb.log(logger.log_current_row)
            logger.dump_tabular()
            game += 1

    print('Average raw returns:', np.mean(env.get_episode_rewards()))
    print(f'Avg returns={avg_ret} over {episodes} episodes')
    env.close()

# ======================================================

wandb.init(project="dqn-eval",
           tags=['BreakoutNoFrameskip-v4', 'DQN'],
           job_type='eval')

monitor_kwargs = dict(
    directory=f'/tmp/openai-gym/{int(time.time())}',
    force=True,
    resume=False,
    video_callable=False
)

torch_load_kwargs = dict(
    f=wandb.restore("pyt_save/model.pt", run_path="frangipane/dqn/4ub3ftgh").name,
    map_location=device,
)

play_game_config = dict(
    torch_load_kwargs=torch_load_kwargs,
    actor_critic=CNNCritic,
    episodes=150,
    render=False,
    logger_kwargs={'exp_name': 'dqn-eval', 'output_dir': wandb.run.dir}
)

if __name__ == '__main__':
    env_id = 'BreakoutNoFrameskip-v4'
    env = make_atari(env_id)
    env = Monitor(env, **monitor_kwargs)
    env = wrap_deepmind(env, frame_stack=True, scale=False)

    if 'Breakout' in env_id:
        if play_game_config['episodes'] % 5 != 0:
            warnings.warn("Consider setting episodes to be a multiple of 5, the number of lives in Breakout")  # noqa
    play_game(env, **play_game_config)

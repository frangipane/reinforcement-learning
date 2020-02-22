"""
Helper script to load a trained model and watch it play for a certain number of episodes, with
option to output recorded video of game play.
"""
import time
import torch
import wandb

from gym.wrappers import Monitor
from atari_wrappers import make_atari, wrap_deepmind

from nnetworks import MLPCritic, CNNCritic


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def play_game(env, torch_load_kwargs={}, actor_critic=CNNCritic, num_episodes=1, render=False):
    ac = actor_critic(env.observation_space, env.action_space)

    # model saved on GPU, load on CPU: https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-across-devices
    ac_saved = torch.load(**torch_load_kwargs)
    ac_saved = ac_saved.to(device)
    ac.q.load_state_dict(ac_saved.q.module.state_dict())

    avg_ret = 0
    for ep in range(num_episodes):
        o, ep_ret, ep_len, d  = env.reset(), 0, 0, False
        while not d:
            if render:
                env.render()
            o = torch.as_tensor(o, dtype=torch.float32).to(device)
            o2, r, d, _ = env.step(ac.act(o))
            ep_ret += r
            ep_len += 1
            o = o2

        print(f"Returns={ep_ret} for episode {ep}")
        avg_ret += (1./(ep+1)) * (ep_ret - avg_ret)

    print(f'Avg returns={avg_ret} over {num_episodes} episodes')

    env.close()


monitor_kwargs = dict(
    directory=f'/tmp/openai-gym/{int(time.time())}',
    force=True,
    resume=False,
    video_callable=lambda episode_id: episode_id % 12 == 0
)
torch_load_kwargs = dict(
    f=wandb.restore("pyt_save/model.pt", run_path="frangipane/dqn/4ub3ftgh").name,
    map_location=device,
)
play_game_config = dict(
    torch_load_kwargs=torch_load_kwargs,
    actor_critic=CNNCritic,
    num_episodes=12,
    render=True,
)

if __name__ == '__main__':
    env = make_atari('BreakoutNoFrameskip-v4')
    env = Monitor(env, **monitor_kwargs)
    env = wrap_deepmind(env, frame_stack=True, scale=False)
    play_game(env, **play_game_config)

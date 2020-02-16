"""Following template laid out in Spinning up in RL, e.g.
https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/ddpg
and
https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/sac

TODOs:
- double check Monitor wrapper params (resume=True or False?)
- Atari environment-specific preprocessing for images, skip frames, concat inputs

"""
import time
from copy import deepcopy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import gym
from gym.wrappers import Monitor
# N.B. to use Monitor, need to have ffmpeg installed,
# e.g. on macOS: brew install ffmpeg

from spinup.utils.logx import EpochLogger
import wandb

from nnetworks import *
from atari_wrappers import make_atari, wrap_deepmind


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n ******** Number of GPUs:", torch.cuda.device_count())


#============================================================================

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    
    Copied from: https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ddpg/ddpg.py#L11,
    modified action buffer for discrete action space.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(size, dtype=np.int32)  # assumes discrete action
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.choice(self.size, size=batch_size, replace=False)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.int32, device=device) if k == 'act'
                else torch.as_tensor(v, dtype=torch.float32, device=device)
                for k,v in batch.items()}


def dqn(env_fn, actor_critic=MLPCritic, replay_size=500,
        seed=0, steps_per_epoch=3000, epochs=5,
        gamma=0.99, lr=0.00025, batch_size=32, start_steps=100, 
        update_after=50, update_every=5,
        epsilon_start=1.0, epsilon_end=0.1, epsilon_step=1e-4,
        target_update_every=1000, num_test_episodes=10, max_ep_len=200,
        record_video=False,
        record_video_every=100, save_freq=50):
    """
    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act``
            method and a ``q`` module.
            The ``act`` method module should accept batches of
            observations as inputs, and ``q`` should accept a batch
            of observations and a batch of actions as inputs. When called,
            ``act`` and ``q`` should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each
                                           | observation.
            ``q``        (batch,)          | Tensor containing the current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

        global_steps (int): Number of steps / frames for training (should be
            greater than update_after!)

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        lr (float): Learning rate.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates.

        epsilon_start (float): Chance to sample a random action when taking an action.
          Epsilon is decayed over time and this is the start value

        epsilon_end (float): The final minimum value of epsilon after decaying is done.

        epsilon_step (float): Reduce epsilon by this amount every step.

        target_update_every (int): Number of steps between updating target network
            parameters, i.e. resetting Q_hat to Q.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout. (Imposed by
           the environment.)

        record_video (bool): Record a video

        record_video_every (int): Record a video every N episodes

        save_freq (int): How often (in terms of gap between epochs) to save
            the current model (value function).
    """
    logger = EpochLogger(exp_name='dqn', output_dir=wandb.run.dir)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()

    if record_video:
        env = Monitor(
            env,
            directory="/tmp/gym-results",
            resume=True,
            video_callable=lambda count: count % record_video_every == 0
        )
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.n  # assumes Discrete space

    # Create critic module and network
    ac = actor_critic(env.observation_space, env.action_space)
    # Set target Q-network parameters theta_tar = theta
    target_q_network = deepcopy(ac.q)

    if torch.cuda.device_count() > 1:
        ac.q = nn.DataParallel(ac.q)
        target_q_network = nn.DataParallel(target_q_network)

    ac.to(device)
    target_q_network.to(device)

    # Freeze target network w.r.t. optimizers
    for p in target_q_network.parameters():
        p.requires_grad = False    

    # function to compute Q-loss
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        
        # Pick out q-values associated with / indexed by the action that was taken
        # for that observation: https://pytorch.org/docs/stable/torch.html#torch.gather.
        # Note index must be of type LongTensor.
        q = torch.gather(ac.q(o), dim=1, index=a.view(-1, 1).long())

        # Bellman backup for Q function
        with torch.no_grad():
            # Targets come from frozen target Q-network
            q_target = torch.max(target_q_network(o2), dim=1).values
            backup = r + (1 - d) * gamma * q_target
            
        # MSE loss against Bellman backup
        # loss_q = ((q - backup)**2).mean()
        loss_q = F.smooth_l1_loss(q[:, 0], backup).mean()
        # TODO: clip Bellman error b/w -1 and 1

        # Useful info for logging
        loss_info = dict(QVals=q.detach().cpu().numpy())

        return loss_q, loss_info

    # Set up optimizer for Q-function
    q_optimizer = torch.optim.Adam(ac.q.parameters(), lr=lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    # function to update parameters in Q
    def update(data):
        q_optimizer.zero_grad()
        loss, loss_info = compute_loss_q(data)
        loss.backward()
        q_optimizer.step()

        logger.store(LossQ=loss.item(), **loss_info)

    def get_action(o, epsilon):
        # greedy epsilon strategy
        if np.random.sample() < epsilon:
            a = env.action_space.sample()
        else:
            a = ac.act(torch.as_tensor(o, dtype=torch.float32, device=device))
        return a

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = test_env.step(get_action(o, 0))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    # main loop: collect experience in env

    # Initialize experience replay buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    epsilon = epsilon_start
    o, ep_ret, ep_len = env.reset(), 0, 0

    for t in range(total_steps):

        if t > start_steps and epsilon > epsilon_end:
            # linearly reduce epsilon
            epsilon -= epsilon_step

        if t > start_steps:
            # epsilon greedy
            a = get_action(o, epsilon)
        else:
            # randomly sample for better exploration before start_steps
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d
        
        # Store transition to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Update the most recent observation.
        o = o2
        
        # End of episode handling
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            left_cnt, right_cnt = 0, 0
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t > update_after and t % update_every == 0:
            minibatch = replay_buffer.sample_batch(batch_size)
            update(data=minibatch)

        # Refresh target Q network
        if t % target_update_every == 0:
            target_q_network.load_state_dict(ac.q.state_dict())
            for p in target_q_network.parameters():
                p.requires_grad = False

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0 and (t+1) > start_steps:
            epoch = (t+1) // steps_per_epoch

            print(f"epsilon: {epsilon}")

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)  # will error if episode lasts longer than epoch since no returns stored
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('QVals', with_min_and_max=True)  # will throw KeyError if update period < epoch period
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.log_tabular('Epsilon', epsilon)
            wandb.log(logger.log_current_row, step=epoch)
            logger.dump_tabular()

    env.close()


# =========== BreakoutNoFrameskip-v0 hyperparameters ===========
# wandb_config = dict(
#     replay_size = 20_000,
#     seed = 0,
#     steps_per_epoch = 640,
#     #epochs = 100,
#     epochs = int(1e7 / 640),
#     gamma = 0.99,
#     lr = 0.00025,
#     batch_size = 64,
#     start_steps = 100,
#     update_after = 100,
#     update_every = 4,
#     epsilon_start = 1.0,
#     epsilon_end = 0.1,
#     epsilon_step = 4e-5,
#     target_update_every = 10_000,
#     max_ep_len = 27000
# )

wandb_config = dict(
    replay_size = 20_000,
    seed = 0,
    steps_per_epoch = 80*32,
    #epochs = 100,
    epochs = int(1e7 / 640),
    gamma = 0.99,
    lr = 0.00025,
    batch_size = 64,
    start_steps = 10_000,
    update_after = 10_000,
    update_every = 4,
    epsilon_start = 1.0,
    epsilon_end = 0.1,
    epsilon_step = 4e-5,
    target_update_every = 10_000,
    max_ep_len = 27000
)

addl_config = dict(
    actor_critic=CNNCritic,
    record_video = False,
    record_video_every = 2000,
    save_freq = 100
)

# =========== CartPole-v1 hyperparameters ===========
# wandb_config = dict(
#     replay_size = 100_000,
#     seed = 0,
#     steps_per_epoch = 640,
#     epochs = 500,
#     gamma = 0.99,
#     lr = 0.00025,
#     batch_size = 64,
#     start_steps = 1000,
#     update_after = 1000,
#     update_every = 1,
#     epsilon_start = 1.0,
#     epsilon_end = 0.1,
#     epsilon_step = 4e-5,
#     target_update_every = 3000
# )

# addl_config = dict(
#     actor_critic=MLPCritic,
#     record_video = False,
#     record_video_every = 2000,
#     save_freq = 100
# )



if __name__ == '__main__':
    #wandb.init(project="dqn", config=wandb_config, tags=['CartPole-v1'])
    #dqn(lambda : gym.make('CartPole-v1'), **wandb_config, **addl_config)

    wandb.init(project="dqn", config=wandb_config, tags=['BreakoutNoFrameskip-v4'])
    env = make_atari('BreakoutNoFrameskip-v4')
    env = wrap_deepmind(env, frame_stack=True, scale=False)
    dqn(lambda: env, **wandb_config, **addl_config)

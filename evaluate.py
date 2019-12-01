import argparse
import torch
import time
import imageio
from pathlib import Path
from torch.autograd import Variable
from utils.make_env import make_env
from algorithms.maddpg import MADDPG
import numpy as np

def run(config):
    print("Evaluating returns for config: %s" % config)
    model_path = (Path('./models') / config.env_id / config.model_name /
                  ('run%i' % config.run_num))
    if config.incremental is not None:
        model_path = model_path / 'incremental' / ('model_ep%i.pt' %
                                                   config.incremental)
    else:
        model_path = model_path / 'model.pt'

    if config.save_gifs:
        gif_path = model_path.parent / 'gifs'
        gif_path.mkdir(exist_ok=True)

    maddpg = MADDPG.init_from_save(model_path)

    # env = make_parallel_env(config.env_id, 1, config.seed, maddpg.discrete_action)
    env = make_env(config.env_id, discrete_action=maddpg.discrete_action)
    maddpg.prep_rollouts(device='cpu')
    ifi = 1 / config.fps  # inter-frame interval

    total_returns = np.zeros(config.n_episodes) # sum of all returns
    agent_returns = np.zeros((config.n_episodes, maddpg.nagents))
    for ep_i in range(config.n_episodes):
        print("Episode %i of %i" % (ep_i + 1, config.n_episodes))
        obs = env.reset()
        if config.save_gifs:
            frames = []
            frames.append(env.render('rgb_array')[0])
        env.render('human')
        # Pick a random subpolicy to execute in this episode
        k = np.random.choice(maddpg.agents[0].K, maddpg.nagents)
        for a in range(maddpg.nagents):

            # if good_agent and ensemble enabled;
            if not (hasattr(env.agents[a], 'adversary') and env.agents[a].adversary) and not config.agent_ens:
                k[a] = 0
            # if adversary and ensemble enabled;
            if hasattr(env.agents[a], 'adversary') and env.agents[a].adversary and not config.adversary_ens:
                k[a] = 0
        for t_i in range(config.episode_length):
            calc_start = time.time()
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(obs[i]).view(1, -1),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]
            # get actions as torch Variables
            voted = maddpg.discrete_action and config.voted_execution
            torch_actions = maddpg.step(torch_obs, explore=False, k = k, voted=voted)
            # convert actions to numpy arrays
            actions = [ac.data.numpy().flatten() for ac in torch_actions]
            obs, rewards, dones, infos = env.step(actions)

            total_returns[ep_i] += config.gamma ** t_i * np.sum(rewards)
            agent_returns[ep_i, :] +=  config.gamma ** t_i * np.array(rewards)
            if config.save_gifs:
                frames.append(env.render('rgb_array')[0])
            calc_end = time.time()
            elapsed = calc_end - calc_start
            if elapsed < ifi:
                time.sleep(ifi - elapsed)
            env.render('human')
        if config.save_gifs:
            gif_num = 0
            while (gif_path / ('%i_%i.gif' % (gif_num, ep_i))).exists():
                gif_num += 1
            imageio.mimsave(str(gif_path / ('%i_%i.gif' % (gif_num, ep_i))),
                            frames, duration=ifi)
    env.close()

    good_indices = []
    adversary_indices = []
    for i in range(len(env.agents)):
        if hasattr(env.agents[i], 'adversary') and env.agents[i].adversary:
            adversary_indices.append(i)
        else:
            good_indices.append(i)
    good_returns = np.sum(agent_returns[:, good_indices], axis=1)
    adversary_returns = np.sum(agent_returns[:, adversary_indices], axis=1)
    return np.mean(total_returns), np.mean(agent_returns, axis=0), \
           np.mean(good_returns, axis=0) / len(good_indices), np.mean(adversary_returns, axis=0) / len(adversary_indices)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_name",
                        help="Name of model")
    parser.add_argument("run_num", default=1, type=int)
    parser.add_argument("--save_gifs", action="store_true",
                        help="Saves gif of each episode into model directory")
    parser.add_argument("--incremental", default=None, type=int,
                        help="Load incremental policy from given episode " +
                             "rather than final policy")
    parser.add_argument("--n_episodes", default=10, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--gamma", default=1.0, type=float)
    parser.add_argument("--voted_execution", action='store_true')

    config = parser.parse_args()

    run(config)
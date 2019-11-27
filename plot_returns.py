from evaluate import run
import argparse
import matplotlib.pyplot as plt
import numpy as np
import copy
import os

def evaluate_returns(config):
    """
    Plot the returns over checkpoints of a trained model
    :param config: The config for the model
    :return: N/A
    """
    checkpoints = np.arange(1, config.trained_episodes, config.step)
    return_eps = np.zeros(checkpoints.shape[0] + 1) # checkpoints + final
    return_good_agents = np.zeros(checkpoints.shape[0] + 1)
    return_agents = []

    # Add flag for disabling gif
    config.save_gifs = False

    for i in range(checkpoints.shape[0]):
        incremental = checkpoints[i]
        run_config = copy.deepcopy(config)
        run_config.incremental = incremental
        total_return, agent_return, good_returns = run(run_config)
        return_eps[i] = total_return
        return_good_agents[i] = good_returns
        return_agents.append(agent_return)

    # Evaluate the final model
    total_return, agent_return, good_returns = run(run_config)
    return_eps[-1] = total_return
    return_good_agents[-1] = good_returns
    return_agents.append(agent_return)

    # Plot the returns
    plot_agents_return(np.array(return_agents), np.append(checkpoints, checkpoints[-1] + config.step), config)
    # plot_return(return_good_agents, np.append(checkpoints, checkpoints[-1] + config.step), config, "GoodAgents")

def plot_agents_return(agents_return, xs, config):
    """
    Plot the returns of the agents
    :param agents_return: The average returns of the agents
    :param xs: The episodes
    :param config: The config
    :return: N/A
    """
    plots = []
    for i in range(agents_return.shape[1]):
        line = plt.plot(xs, agents_return[:, i], label='Agent %d' % i)
        plots.append(line)
    plt.legend()

    plt.title('Agents return')

    fig_dir = './plots/' + config.env_id + '/' + config.model_name
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    plt.savefig(fig_dir + '/agents_return.png')
    plt.show()

def plot_return(returns, xs, config, label):
    """
    Plot the returns;
    :param returns: The returns for which to plot
    :param xs: The episodes
    :param config: configuration for evaluation
    :return: N/A
    """
    plots = []
    l1 = plt.plot(xs, returns, label=label)
    plots.append(l1)
    plt.legend()

    plt.title('Average return')

    fig_dir = './plots/' + config.env_id  + '/' + config.model_name
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    plt.savefig(fig_dir + '/%s.png' % label)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_name",
                        help="Name of model")
    parser.add_argument("run_num", default=1, type=int)
    parser.add_argument("--n_episodes", default=10, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--gamma", default=1.0, type=float)
    parser.add_argument("--trained_episodes", default=25000, type=int)
    parser.add_argument("--step", default=1000, type=int)
    parser.add_argument("--n_adversary", default=0, type=int)

    config = parser.parse_args()

    evaluate_returns(config)
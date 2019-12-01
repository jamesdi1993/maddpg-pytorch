from ast import literal_eval
from deprecated import deprecated
from evaluate import run
import argparse
import matplotlib.pyplot as plt
import numpy as np
import copy
import os

def build_common_config(config, model_config):
    """
    Build a common run configuration for each model
    :param config: A Plot Generation Config
    :return: A common run config
    """
    model_name, run_num, agent_ens, adversary_ens, voted = model_config
    config_base = copy.deepcopy(config)
    config_base.save_gifs = False
    config_base.model_name = model_name
    config_base.run_num = run_num
    config_base.agent_ens = agent_ens
    config_base.adversary_ens = adversary_ens
    config_base.voted_execution = voted
    return config_base

def build_label(config):
    """
    Build a label from config
    :param config: A run config
    :return: A string label, useful for saving plots.
    """
    label = config.model_name
    if config.agent_ens:
        label += '_agent_ens'
    if config.adversary_ens:
        label += '_adversary_ens'
    if config.voted_execution:
        label += '_voted'
    return label

def prepare_model_configs(config):
    """
    Prepare configs from plot configuration
    :param config: a plot configuration
    :return: a list of list of run configuration and checkpoints;
    """
    models = literal_eval(config.models)
    model_configs = []
    checkpoints = np.arange(1, config.trained_episodes, config.step)
    for model_config in models:
        run_configs = []
        config_base = build_common_config(config, model_config)
        for i in range(checkpoints.shape[0]):
            run_config = copy.deepcopy(config_base)
            incremental = checkpoints[i]
            run_config.incremental = incremental
            run_configs.append(run_config)

        # Add final model config
        final_config = copy.deepcopy(config_base)
        final_config.incremental = None
        run_configs.append(final_config)

        # Add the run_configs for the model
        model_configs.append(run_configs)
    # Add final timestep into the config;
    checkpoints = np.append(checkpoints, checkpoints[-1] + config.step)
    return model_configs, checkpoints

def evaluate_model_returns(config):
    """
    Evaluate the return for one or a few models.
    :param config: The configuration;
    :return: N/A
    """
    returns = []
    model_configs, checkpoints = prepare_model_configs(config)
    labels = []
    for run_configs in model_configs:
        print("Evaluating returns for model: %s" % run_configs[0].model_name)
        _, _, return_good_agents, return_adversary = evaluate_run_returns(run_configs)
        returns.append(return_good_agents)
        label = build_label(run_configs[0])
        labels.append(label)
    plot_returns(np.array(returns), checkpoints, config, labels)

def evaluate_run_returns(run_configs):
    """
    Evaluate the returns over a number of run_configs;
    :return:
    """
    return_eps = np.zeros(len(run_configs))  # checkpoints + final
    return_good_agents = np.zeros(len(run_configs))
    return_adversary_agents = np.zeros(len(run_configs))
    return_agents = []
    for i in range(len(run_configs)):
        run_config = run_configs[i]
        total_return, agent_return, good_returns, adversary_returns = run(run_config)
        return_eps[i] = total_return
        return_good_agents[i] = good_returns
        return_adversary_agents[i] = adversary_returns
        return_agents.append(agent_return)
    return return_eps, return_agents, return_good_agents, return_adversary_agents

@deprecated
def evaluate_returns(config):
    """
    Plot the returns over checkpoints of a trained model
    :param config: The config for the model
    :return: N/A
    """
    checkpoints = np.arange(1, config.trained_episodes, config.step)
    return_eps = np.zeros(checkpoints.shape[0] + 1) # checkpoints + final
    return_good_agents = np.zeros(checkpoints.shape[0] + 1)
    return_adversary_agents = np.zeros(checkpoints.shape[0] + 1)
    return_agents = []

    # Add flag for disabling gif
    config.save_gifs = False

    for i in range(checkpoints.shape[0]):
        incremental = checkpoints[i]
        run_config = copy.deepcopy(config)
        run_config.incremental = incremental
        total_return, agent_return, good_returns, adversary_returns = run(run_config)
        return_eps[i] = total_return
        return_good_agents[i] = good_returns
        return_adversary_agents[i] = adversary_returns
        return_agents.append(agent_return)

    # Evaluate the final model
    total_return, agent_return, good_returns, adversary_returns = run(run_config)
    return_eps[-1] = total_return
    return_good_agents[-1] = good_returns
    return_adversary_agents[-1] = adversary_returns
    return_agents.append(agent_return)

    # Plot the returns
    # plot_agents_return(np.array(return_agents), np.append(checkpoints, checkpoints[-1] + config.step), config)
    plot_return(return_adversary_agents, np.append(checkpoints, checkpoints[-1] + config.step), config, "Adversary")

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

@deprecated
# Use plot_returns instead;
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

'''
Sample Usage:
plot_model_returns(G_i, xs, config, ['Ensemble'];
plot_model_returns(returns, xs, config, ['Ensemble', 'Ensemble Voted'];
'''
def plot_returns(returns, xs, config, labels):
    """
    Plot the returns for multiple models
    :param returns: The returns for which to plot
    :param xs: The episodes
    :param config: configuration for evaluation
    :return: N/A
    """
    plots = []
    for i in range(returns.shape[0]):
        line = plt.plot(xs, returns[i, :], label=labels[i])
        plots.append(line)
    plt.legend()

    plt.title('Average return')
    fig_dir = './plots/' + config.env_id
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    file_name = "_".join(labels)
    plt.savefig(fig_dir + '/%s.png' % file_name)
    plt.show()

# Example usage: python plot_returns.py simple_tag --models "[('maddpg_vs_ddpg',1, False, False, False),('maddpg_vs_maddpg',1, True, False, True)]" --n_episodes 10
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    # Use space as delimiter to separate out the models
    parser.add_argument('--models', help="Models to compare with '(model_name, run_id, agent_ens, adversary_ens, voted_evaluation), ...'" + \
                                         "e.g '('Ensemble-Voted',1, True, False, True),('Ensemble-Random',2, True, False, False)'", type=str)
    parser.add_argument("--n_episodes", default=10, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--gamma", default=1.0, type=float)
    parser.add_argument("--trained_episodes", default=25000, type=int)
    parser.add_argument("--step", default=1000, type=int)
    parser.add_argument("--n_adversary", default=0, type=int)

    config = parser.parse_args()

    evaluate_model_returns(config)
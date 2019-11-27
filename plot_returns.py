from evaluate import run
from pathlib import Path
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

    # Add flag for disabling gif
    config.save_gifs = False

    for i in range(checkpoints.shape[0]):
        incremental = checkpoints[i]
        run_config = copy.deepcopy(config)
        run_config.incremental = incremental
        print("The config for evaluation is: %s" % run_config)
        total_return, _ = run(run_config)
        return_eps[i] = total_return

    total_return, _ = run(run_config)
    return_eps[-1] = total_return # Evaluate the final model
    plot_return(return_eps, np.append(checkpoints, checkpoints[-1] + config.step), config)

def plot_return(G, xs, config):
    plt.plot(xs, G)
    plt.title('Average return')

    fig_dir = './plots/' + config.env_id  + '/' + config.model_name
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    plt.savefig(fig_dir + '/returns.png')
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

    config = parser.parse_args()

    evaluate_returns(config)
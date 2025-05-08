import gymnasium as gym
import retro
import json
import numpy as np
import matplotlib.pyplot as plt
import os

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import results_plotter
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import Monitor

from retro_wrapper import RetroArcade

from tqdm import tqdm


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}"
                    )

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path)

        return True


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def plot_results(log_folder, title="Learning Curve", game=None):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), "timesteps")
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y) :]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")
    plt.savefig(log_folder + game + "_results.png")
    # plt.show()


Mario1GameStarts = {
    "SuperMarioBros-Nes": [
        "Level1-1.state",
        #  "Level2-1.state",
    ],
}

# Mario2GameStarts = {
#     "SuperMarioBros2Japan-Nes": [
#         "Level1-1.state",
#         "Level1-2.state",
#         "Level2-1.state",
#         "Level3-1.state",
#         "Level4-1.state",
#         "Level5-1.state",
#         "Level6-1.state",
#         "Level6-2.state",
#         "Level7-1.state",
#         "Level8-1.state",
#     ],
# }

if __name__ == "__main__":

    s1_test_games = json.load(open("retro_game_sets/s1_test.json"))
    total_timesteps = int(1e6)
    game_level_list = []

    for game, levels in s1_test_games.items():
        for level in levels:
            game_level_list.append((game, level))

    game_level_list = sorted(game_level_list)
    print(len(game_level_list))
    # for i, game_start_pair in enumerate(game_level_list):
    #     if i % 2 == 1:
    #         game, levels = game_start_pair
    #         game_start_dict = {game: levels}
    #         env = RetroArcade(game_start_dict=game_start_dict, use_discrete_actions=True)
    #         log_dir = "./dqn_logs/" + game + "/"
    #         env = Monitor(env, log_dir)

    #         callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=log_dir)

    #         model = DQN("MlpPolicy", env, verbose=1, device="cpu")

    #         model.learn(total_timesteps=total_timesteps, log_interval=1, callback=callback, progress_bar=True)

    #         env.close()
    #         results_plotter.plot_results([log_dir], total_timesteps, results_plotter.X_TIMESTEPS, "DQN " + game)
    #         #plt.show()
    #         plot_results(log_dir, f"DQN {game} Reward Curve", game)

    env = RetroArcade(game_start_dict=Mario1GameStarts, use_discrete_actions=True)
    log_dir = "./dqn_logs/" + "Mario" + "/"
    env = Monitor(env, log_dir)
    game = "Mario"
    callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=log_dir)

    model = DQN("MlpPolicy", env, verbose=1, device="cpu")

    model.learn(
        total_timesteps=total_timesteps,
        log_interval=1,
        callback=callback,
        progress_bar=True,
    )

    env.close()
    results_plotter.plot_results(
        [log_dir], total_timesteps, results_plotter.X_TIMESTEPS, "DQN " + game
    )
    #         #plt.show()
    plot_results(log_dir, f"DQN {game} Reward Curve", game)

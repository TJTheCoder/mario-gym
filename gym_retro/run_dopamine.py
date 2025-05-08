import numpy as np
import os
from dopamine.discrete_domains import run_experiment
from dopamine.colab import utils as colab_utils
from absl import flags
import gin.tf

import seaborn as sns
import matplotlib.pyplot as plt

import json
import sys

BASE_PATH = "tmp/dopamine_run"  # @param


if __name__ == "__main__":

    s1_test_games = json.load(open("retro_game_sets/s1_test.json"))
    total_timesteps = int(1e6)
    game_level_list = []

    for game, levels in s1_test_games.items():
        for level in levels:
            game_level_list.append((game, level))

    game_level_list = sorted(game_level_list)

    experiment_index = 0
    if len(sys.argv) > 1:
        try:
            experiment_index = int(sys.argv[1])
        except:
            raise TypeError("You must specify int type for experiment index")
    print(experiment_index)

    for i, game_start_pair in enumerate(game_level_list):
        if i % 8 == experiment_index:
            game, level = game_start_pair
            game_start_dict = {game: [level]}
            game_level_str = f"{game}_{level}"
            RAINBOW_PATH = os.path.join(BASE_PATH, f"rainbow_{game_level_str}")

            rainbow_config = f"""
            # Hyperparameters follow the classic Nature DQN, but we modify as necessary to
            # match those used in Rainbow (Hessel et al., 2018), to ensure apples-to-apples
            # comparison.
            import dopamine.agents.rainbow.rainbow_agent
            import dopamine.discrete_domains.gym_lib
            import dopamine.discrete_domains.run_experiment
            import dopamine.replay_memory.prioritized_replay_buffer
            import gin.tf.external_configurables

            RainbowAgent.num_atoms = 51
            RainbowAgent.vmax = 10.
            RainbowAgent.gamma = 0.99
            RainbowAgent.update_horizon = 20 # 3
            RainbowAgent.min_replay_history = 1600  # agent steps
            RainbowAgent.update_period = 4 # 4
            RainbowAgent.target_update_period = 2000  # agent steps
            RainbowAgent.epsilon_train = 0.01
            RainbowAgent.epsilon_eval = 0.001
            RainbowAgent.epsilon_decay_period = 250000  # agent steps
            RainbowAgent.replay_scheme = 'prioritized'
            RainbowAgent.tf_device = '/cpu:*' #'/gpu:0'  # use '/cpu:*' for non-GPU version
            RainbowAgent.optimizer = @tf.train.AdamOptimizer()

            # Note these parameters are different from C51's.
            tf.train.AdamOptimizer.learning_rate = 0.0001 #0.0000625
            tf.train.AdamOptimizer.epsilon = 0.00015

            create_gym_environment.environment_name = {game_start_dict} # 'Mario' #'Mario'
            create_gym_environment.version = 'v1'
            # atari_lib.create_atari_environment.game_name = "Mario"
            # Sticky actions with probability 0.25, as suggested by (Machado et al., 2017).
            # gym_lib.create_atari_environment.sticky_actions = False # True
            create_agent.agent_name = 'rainbow'
            Runner.create_environment_fn = @gym_lib.create_gym_environment
            Runner.num_iterations = 10
            Runner.training_steps = 100000  # agent steps
            Runner.evaluation_steps = 0  # agent steps
            Runner.max_steps_per_episode = 108000  # agent steps

            WrappedPrioritizedReplayBuffer.replay_capacity = 1000000
            WrappedPrioritizedReplayBuffer.batch_size = 32
            """

            gin.parse_config(rainbow_config, skip_unknown=False)
            rainbow_runner = run_experiment.create_runner(
                RAINBOW_PATH, schedule="continuous_train"
            )
            print("Will train RAINBOW agent, please be patient, may be a while...")
            rainbow_runner.run_experiment()
            print("Done training!")

            data = colab_utils.read_experiment(
                RAINBOW_PATH, verbose=True, summary_keys=["train_episode_returns"]
            )
            data["agent"] = "RAINBOW"
            data["run"] = 1

            fig, ax = plt.subplots(figsize=(16, 8))
            sns.lineplot(
                x="iteration", y="train_episode_returns", hue="agent", data=data, ax=ax
            )
            plt.title(f"Rainbow DQN {game_level_str} Results")
            plt.savefig(f"{game_level_str}_rainbow_results.png")

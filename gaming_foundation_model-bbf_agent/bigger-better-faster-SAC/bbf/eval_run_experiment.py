# coding=utf-8

#greedy_frac = 0.5
greedy_frac = 0
import random

import functools
import os
import sys
import time
import gym
import cv2
from PIL import Image

from absl import logging
import gin
import jax
import numpy as np
import h5py
from datetime import datetime

atari_human_scores = {
    'Alien': 7127.7,
    'Amidar': 1719.5,
    'Assault': 742.0,
    'Asterix': 8503.3,
    'Asteroids': 47388.7,
    'Atlantis': 29028.1,
    'BankHeist': 753.1,
    'BattleZone': 37187.5,
    'BeamRider': 16926.5,
    'Berzerk': 2630.4,
    'Bowling': 160.7,
    'Boxing': 12.1,
    'Breakout': 30.5,
    'Centipede': 12017.0,
    'ChopperCommand': 7387.8,
    'CrazyClimber': 35829.4,
    'DemonAttack': 1971.0,
    'DoubleDunk': -16.4,
    'Enduro': 860.5,
    'FishingDerby': -38.7,
    'Freeway': 29.6,
    'Frostbite': 4334.7,
    'Gopher': 2412.5,
    'Gravitar': 3351.4,
    'Hero': 30826.4,
    'IceHockey': 0.9,
    'Jamesbond': 302.8,
    'Kangaroo': 3035.0,
    'Krull': 2665.5,
    'KungFuMaster': 22736.3,
    'MontezumaRevenge': 4753.3,
    'MsPacman': 6951.6,
    'NameThisGame': 8049.0,
    'Phoenix': 7242.6,
    'Pitfall': 6463.7,
    'Pong': 14.6,
    'PrivateEye': 69571.3,
    'Qbert': 13455.0,
    'Riverraid': 17118.0,
    'RoadRunner': 7845.0,
    'Robotank': 11.9,
    'Seaquest': 42054.7,
    'Skiing': -4336.9,
    'Solaris': 12326.7,
    'SpaceInvaders': 1668.7,
    'StarGunner': 10250.0,
    'Tennis': -8.3,
    'TimePilot': 5229.2,
    'Tutankham': 167.6,
    'UpNDown': 11693.2,
    'Venture': 1187.5,
    'VideoPinball': 17667.9,
    'WizardOfWor': 4756.5,
    'YarsRevenge': 54576.9,
    'Zaxxon': 9173.3,
}

atari_random_scores = {
    'Alien': 227.8,
    'Amidar': 5.8,
    'Assault': 222.4,
    'Asterix': 210.0,
    'Asteroids': 719.1,
    'Atlantis': 12850.0,
    'BankHeist': 14.2,
    'BattleZone': 2360.0,
    'BeamRider': 363.9,
    'Berzerk': 123.7,
    'Bowling': 23.1,
    'Boxing': 0.1,
    'Breakout': 1.7,
    'Centipede': 2090.9,
    'ChopperCommand': 811.0,
    'CrazyClimber': 10780.5,
    'Defender': 2874.5,
    'DemonAttack': 152.1,
    'DoubleDunk': -18.6,
    'Enduro': 0.0,
    'FishingDerby': -91.7,
    'Freeway': 0.0,
    'Frostbite': 65.2,
    'Gopher': 257.6,
    'Gravitar': 173.0,
    'Hero': 1027.0,
    'IceHockey': -11.2,
    'Jamesbond': 29.0,
    'Kangaroo': 52.0,
    'Krull': 1598.0,
    'KungFuMaster': 258.5,
    'MontezumaRevenge': 0.0,
    'MsPacman': 307.3,
    'NameThisGame': 2292.3,
    'Phoenix': 761.4,
    'Pitfall': -229.4,
    'Pong': -20.7,
    'PrivateEye': 24.9,
    'Qbert': 163.9,
    'Riverraid': 1338.5,
    'RoadRunner': 11.5,
    'Robotank': 2.2,
    'Seaquest': 68.4,
    'Skiing': -17098.1,
    'Solaris': 1236.3,
    'SpaceInvaders': 148.0,
    'StarGunner': 664.0,
    'Surround': -10.0,
    'Tennis': -23.8,
    'TimePilot': 3568.0,
    'Tutankham': 11.4,
    'UpNDown': 533.4,
    'Venture': 0.0,
    'VideoPinball': 0.0,
    'WizardOfWor': 563.5,
    'YarsRevenge': 3092.9,
    'Zaxxon': 32.5,
}
atari_random_scores = {k.lower(): v for k, v in atari_random_scores.items()}
atari_human_scores = {k.lower(): v for k, v in atari_human_scores.items()}

# Add Mario-specific scores for normalization
mario_random_scores = {
    'supermariobros-nes': 0.0,  # Random agent typically gets 0
    'supermariobros-nes-level1-1': 0.0
}

mario_human_scores = {
    'supermariobros-nes': 5000.0,  # Approximate human score
    'supermariobros-nes-level1-1': 5000.0
}

def normalize_score(ret, game):
    """Normalize score between random and human performance."""
    game = game.lower().replace('_', '').replace(' ', '')
    logging.info('Normalizing score for game: %s, score: %.2f', game, ret)
    
    # Try to get scores from the appropriate dictionary
    random_score = mario_random_scores.get(game, atari_random_scores.get(game, 0.0))
    human_score = mario_human_scores.get(game, atari_human_scores.get(game, 5000.0))
    
    logging.info('Using random score: %.2f, human score: %.2f', random_score, human_score)
    
    if human_score == random_score:
        return 0.0 if ret <= random_score else 1.0
    
    normalized = (ret - random_score) / (human_score - random_score)
    logging.info('Normalized score: %.2f', normalized)
    return normalized

def create_env_wrapper(create_env_fn):

    def inner_create(*args, **kwargs):
        env = create_env_fn(*args, **kwargs)
        env.cum_length = 0
        env.cum_reward = 0
        return env

    return inner_create


@gin.configurable
def create_atari_environment(game_name=None, sticky_actions=True):
    assert game_name is not None
    game_version = 'v0' if sticky_actions else 'v4'
    full_game_name = '{}NoFrameskip-{}'.format(game_name, game_version)
    env = gym.make(full_game_name)
    # Strip out the TimeLimit wrapper from Gym, which caps us at 100k frames. We
    # handle this time limit internally instead, which lets us cap at 108k frames
    # (30 minutes). The TimeLimit wrapper also plays poorly with saving and
    # restoring states.
    env = env.env
    env = AtariPreprocessing(env)
    return env


@gin.configurable
class AtariPreprocessing(object):
    """A class implementing image preprocessing for Atari 2600 agents.

  Specifically, this provides the following subset from the JAIR paper
  (Bellemare et al., 2013) and Nature DQN paper (Mnih et al., 2015):

    * Frame skipping (defaults to 4).
    * Terminal signal when a life is lost (off by default).
    * Grayscale and max-pooling of the last two frames.
    * Downsample the screen to a square image (defaults to 84x84).

  More generally, this class follows the preprocessing guidelines set down in
  Machado et al. (2018), "Revisiting the Arcade Learning Environment:
  Evaluation Protocols and Open Problems for General Agents".
  """

    def __init__(self,
                 environment,
                 frame_skip=4,
                 terminal_on_life_loss=False,
                 screen_size=84):
        """Constructor for an Atari 2600 preprocessor.

    Args:
      environment: Gym environment whose observations are preprocessed.
      frame_skip: int, the frequency at which the agent experiences the game.
      terminal_on_life_loss: bool, If True, the step() method returns
        is_terminal=True whenever a life is lost. See Mnih et al. 2015.
      screen_size: int, size of a resized Atari 2600 frame.

    Raises:
      ValueError: if frame_skip or screen_size are not strictly positive.
    """
        if frame_skip <= 0:
            raise ValueError(
                'Frame skip should be strictly positive, got {}'.format(
                    frame_skip))
        if screen_size <= 0:
            raise ValueError(
                'Target screen size should be strictly positive, got {}'.format(
                    screen_size))

        self.environment = environment
        self.terminal_on_life_loss = terminal_on_life_loss
        self.frame_skip = frame_skip
        self.screen_size = screen_size

        obs_dims = self.environment.observation_space
        # Stores temporary observations used for pooling over two successive
        # frames.
        self.screen_buffer = [
            np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8),
            np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8)
        ]

        self.game_over = False
        self.lives = 0  # Will need to be set by reset().

    @property
    def observation_space(self):
        # Return the observation space adjusted to match the shape of the processed
        # observations.
        return Box(low=0,
                   high=255,
                   shape=(self.screen_size, self.screen_size, 1),
                   dtype=np.uint8)

    @property
    def action_space(self):
        return self.environment.action_space

    @property
    def reward_range(self):
        return self.environment.reward_range

    @property
    def metadata(self):
        return self.environment.metadata

    def close(self):
        return self.environment.close()

    def reset(self):
        """Resets the environment.

    Returns:
      observation: numpy array, the initial observation emitted by the
        environment.
    """
        self.environment.reset()
        self.lives = self.environment.ale.lives()
        self._fetch_grayscale_observation(self.screen_buffer[0])
        self.screen_buffer[1].fill(0)
        return self._pool_and_resize()

    def render(self, mode):
        """Renders the current screen, before preprocessing.

    This calls the Gym API's render() method.

    Args:
      mode: Mode argument for the environment's render() method.
        Valid values (str) are:
          'rgb_array': returns the raw ALE image.
          'human': renders to display via the Gym renderer.

    Returns:
      if mode='rgb_array': numpy array, the most recent screen.
      if mode='human': bool, whether the rendering was successful.
    """
        return self.environment.render(mode)

    def step(self, action):
        """Applies the given action in the environment.

    Remarks:

      * If a terminal state (from life loss or episode end) is reached, this may
        execute fewer than self.frame_skip steps in the environment.
      * Furthermore, in this case the returned observation may not contain valid
        image data and should be ignored.

    Args:
      action: The action to be executed.

    Returns:
      observation: numpy array, the observation following the action.
      reward: float, the reward following the action.
      is_terminal: bool, whether the environment has reached a terminal state.
        This is true when a life is lost and terminal_on_life_loss, or when the
        episode is over.
      info: Gym API's info data structure.
    """
        accumulated_reward = 0.

        for time_step in range(self.frame_skip):
            # We bypass the Gym observation altogether and directly fetch the
            # grayscale image from the ALE. This is a little faster.
            _, reward, game_over, info = self.environment.step(action)
            accumulated_reward += reward

            if self.terminal_on_life_loss:
                new_lives = self.environment.ale.lives()
                is_terminal = game_over or new_lives < self.lives
                self.lives = new_lives
            else:
                is_terminal = game_over

            # We max-pool over the last two frames, in grayscale.
            if time_step >= self.frame_skip - 2:
                t = time_step - (self.frame_skip - 2)
                self._fetch_grayscale_observation(self.screen_buffer[t])

            if is_terminal:
                break

        # Pool the last two observations.
        observation = self._pool_and_resize()

        self.game_over = game_over
        return observation, accumulated_reward, is_terminal, info

    def _fetch_grayscale_observation(self, output):
        """Returns the current observation in grayscale.

    The returned observation is stored in 'output'.

    Args:
      output: numpy array, screen buffer to hold the returned observation.

    Returns:
      observation: numpy array, the current observation in grayscale.
    """
        self.environment.ale.getScreenGrayscale(output)
        return output

    def _pool_and_resize(self):
        """Transforms two frames into a Nature DQN observation.

    For efficiency, the transformation is done in-place in self.screen_buffer.

    Returns:
      transformed_screen: numpy array, pooled, resized screen.
    """
        # Pool if there are enough screens to do so.
        if self.frame_skip > 1:
            np.maximum(self.screen_buffer[0],
                       self.screen_buffer[1],
                       out=self.screen_buffer[0])

        transformed_image = cv2.resize(self.screen_buffer[0],
                                       (self.screen_size, self.screen_size),
                                       interpolation=cv2.INTER_AREA)
        int_image = np.asarray(transformed_image, dtype=np.uint8)
        return np.expand_dims(int_image, axis=2)


@gin.configurable
def create_retro_environment(game_name=None, state_name=None):
    """Creates a stable-retro environment.
    
    Args:
        game_name: str, name of the game (e.g. 'SuperMarioBros-Nes')
        state_name: str, name of the state/level (e.g. 'Level1-1')
    """
    # If game_name or state_name are not provided, try to get them from gin config
    if game_name is None:
        game_name = gin.query_parameter('create_retro_environment.game_name')
    if state_name is None:
        state_name = gin.query_parameter('create_retro_environment.state_name')
    
    assert game_name is not None, "game_name must be provided either as argument or in gin config"
    assert state_name is not None, "state_name must be provided either as argument or in gin config"
    
    import retro
    # Disable rendering by setting render_mode to None
    env = retro.make(game=game_name, state=state_name, render_mode=None)
    env = RetroPreprocessing(env)
    return env

@gin.configurable
class RetroPreprocessing(object):
    """A class implementing image preprocessing for Retro environments."""

    def __init__(self, environment, frame_skip=4, screen_size=84):
        """Constructor for a Retro preprocessor.

        Args:
            environment: Gym environment whose observations are preprocessed.
            frame_skip: int, the frequency at which the agent experiences the game.
            screen_size: int, size of a resized frame.
        """
        if frame_skip <= 0:
            raise ValueError('Frame skip should be strictly positive, got {}'.format(frame_skip))
        if screen_size <= 0:
            raise ValueError('Target screen size should be strictly positive, got {}'.format(screen_size))

        self.environment = environment
        self.frame_skip = frame_skip
        self.screen_size = screen_size

        # Get the number of buttons from the environment
        self.num_buttons = self.environment.num_buttons
        logging.info('RetroPreprocessing initialized with %d buttons', self.num_buttons)

        obs_dims = self.environment.observation_space
        logging.info('Observation space dimensions: %s', obs_dims.shape)
        self.screen_buffer = [
            np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8),
            np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8)
        ]

        # Track game over state
        self.game_over = False
        self.cum_length = 0
        self.cum_reward = 0
        logging.info('Game state tracking initialized')

    @property
    def observation_space(self):
        return Box(low=0, high=255, shape=(self.screen_size, self.screen_size, 1), dtype=np.uint8)

    @property
    def action_space(self):
        # Return a discrete action space with 2^num_buttons actions
        return gym.spaces.Discrete(2 ** self.num_buttons)

    @property
    def reward_range(self):
        return self.environment.reward_range

    @property
    def metadata(self):
        return self.environment.metadata

    def close(self):
        return self.environment.close()

    def reset(self):
        """Resets the environment."""
        logging.info('Resetting environment')
        self.environment.reset()
        self._fetch_grayscale_observation(self.screen_buffer[0])
        self.screen_buffer[1].fill(0)
        self.game_over = False
        self.cum_length = 0
        self.cum_reward = 0
        logging.info('Environment reset complete')
        return self._pool_and_resize()

    def _int_to_action(self, action):
        """Convert an integer action to a binary action array."""
        # Convert the integer to a binary array
        binary = np.zeros(self.num_buttons, dtype=np.uint8)
        for i in range(self.num_buttons):
            binary[i] = (action >> i) & 1
        return binary

    def _fetch_grayscale_observation(self, output):
        """Fetches the grayscale observation from the environment."""
        # Get the screen data directly from the emulator
        screen = self.environment.em.get_screen()
        # Convert to grayscale
        observation = np.mean(screen, axis=2)
        output[:] = observation

    def _pool_and_resize(self):
        """Pools and resizes the observation."""
        # Max pooling over the last two frames
        pooled = np.maximum(self.screen_buffer[0], self.screen_buffer[1])
        # Resize to target size
        resized = np.array(Image.fromarray(pooled).resize(
            (self.screen_size, self.screen_size), Image.BILINEAR))
        return resized.reshape(self.screen_size, self.screen_size, 1)

    def step(self, action):
        """Steps through the environment."""
        # Convert the integer action to a binary action array
        binary_action = self._int_to_action(action)
        logging.debug('Action: %d -> Binary: %s', action, binary_action)
        
        total_reward = 0.0
        done = False
        info = {}
        
        for i in range(self.frame_skip):
            # Get the step result
            step_result = self.environment.step(binary_action)
            
            # Handle different return value formats
            if len(step_result) == 4:
                observation, reward, done, info = step_result
            elif len(step_result) == 5:
                observation, reward, done, truncated, info = step_result
                done = done or truncated
            else:
                raise ValueError(f"Unexpected number of return values from step: {len(step_result)}")
            
            total_reward += reward
            self.cum_reward += reward
            self.cum_length += 1
            
            logging.debug('Step %d/%d: reward=%.2f, total=%.2f, length=%d', 
                         i+1, self.frame_skip, reward, self.cum_reward, self.cum_length)
            
            if done:
                self.game_over = True
                logging.info('Episode ended: total_reward=%.2f, length=%d', 
                           self.cum_reward, self.cum_length)
                break
            self._fetch_grayscale_observation(self.screen_buffer[1])
            self.screen_buffer[0], self.screen_buffer[1] = self.screen_buffer[1], self.screen_buffer[0]
        
        return self._pool_and_resize(), total_reward, done, info

    def render(self, mode):
        """Renders the current screen."""
        if mode == 'rgb_array':
            return self._pool_and_resize()
        return None


class EpisodeRecorder:
    """Records episode interactions to HDF5 files, one file per episode."""
    
    def __init__(self, save_dir='episode_recordings'):
        """Initialize the episode recorder.
        
        Args:
            save_dir: Directory to save HDF5 files
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.current_episode = 0
        self.current_step = 0
        self.current_file = None
        
        # Create first episode file
        self._create_new_episode_file(0)
        
        logging.info('Created episode recorder in directory %s', save_dir)
    
    def _create_new_episode_file(self, episode_num):
        """Create a new HDF5 file for an episode.
        
        Args:
            episode_num: Episode number
        """
        if self.current_file is not None:
            self.current_file.close()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(self.save_dir, f'episode_{episode_num}_{timestamp}.h5')
        self.current_file = h5py.File(filename, 'w')
        self.current_step = 0
        logging.info('Created new episode file: %s', filename)
    
    def record_step(self, episode_num, step_num, observation, action, reward, done, info):
        """Record a single step of an episode.
        
        Args:
            episode_num: Episode number
            step_num: Step number within episode
            observation: Current observation
            action: Action taken
            reward: Reward received
            done: Whether episode is done
            info: Additional info
        """
        try:
            # Create new file if this is a new episode
            if episode_num != self.current_episode:
                self._create_new_episode_file(episode_num)
                self.current_episode = episode_num
            
            # Convert data to numpy arrays
            observation = np.asarray(observation, dtype=np.uint8)
            action = np.asarray(action, dtype=np.int32)
            reward = np.asarray(reward, dtype=np.float32)
            done = np.asarray(done, dtype=np.bool_)
            
            # Create or resize datasets
            if self.current_step == 0:
                # First step - create datasets
                self.current_file.create_dataset('observations', 
                                               shape=(1,) + observation.shape,
                                               maxshape=(None,) + observation.shape,
                                               dtype=np.uint8)
                self.current_file.create_dataset('actions',
                                               shape=(1,) + action.shape,
                                               maxshape=(None,) + action.shape,
                                               dtype=np.int32)
                self.current_file.create_dataset('rewards',
                                               shape=(1,),
                                               maxshape=(None,),
                                               dtype=np.float32)
                self.current_file.create_dataset('dones',
                                               shape=(1,),
                                               maxshape=(None,),
                                               dtype=np.bool_)
            else:
                # Resize datasets to accommodate new step
                for name in ['observations', 'actions', 'rewards', 'dones']:
                    self.current_file[name].resize(self.current_step + 1, axis=0)
            
            # Store the data
            self.current_file['observations'][self.current_step] = observation
            self.current_file['actions'][self.current_step] = action
            self.current_file['rewards'][self.current_step] = reward
            self.current_file['dones'][self.current_step] = done
            
            # Store info as attributes
            for key, value in info.items():
                if isinstance(value, (int, float, str)):
                    self.current_file.attrs[f'info_{key}'] = value
            
            self.current_step += 1
            self.current_file.flush()
            
        except Exception as e:
            logging.error('Failed to record step: %s', str(e))
            raise
    
    def close(self):
        """Close the current HDF5 file."""
        try:
            if self.current_file is not None:
                self.current_file.close()
                self.current_file = None
                logging.info('Closed episode recorder')
        except Exception as e:
            logging.error('Failed to close HDF5 file: %s', str(e))
            raise

@gin.configurable
class Runner(object):
    """Base class for running experiments."""

    def __init__(
        self,
        create_agent_fn,
        create_environment_fn=create_retro_environment,
        checkpoint_file_prefix='ckpt',
        logging_file_prefix='log',
        log_every_n=1,
        num_iterations=200,
        training_steps=250000,
        evaluation_steps=125000,
        max_steps_per_episode=27000,
        clip_rewards=True,
        explore_end_steps=None,
    ):
        """Initialize the runner.

        Args:
            create_agent_fn: Function that creates an agent.
            create_environment_fn: Function that creates an environment.
            checkpoint_file_prefix: Prefix for checkpoint files.
            logging_file_prefix: Prefix for logging files.
            log_every_n: Log every n steps.
            num_iterations: Number of training iterations.
            training_steps: Number of training steps per iteration.
            evaluation_steps: Number of evaluation steps per iteration.
            max_steps_per_episode: Maximum steps per episode.
            clip_rewards: Whether to clip rewards.
            explore_end_steps: Number of steps after which exploration ends.
        """
        # Store basic configuration
        self._logging_file_prefix = logging_file_prefix
        self._log_every_n = log_every_n
        self._num_iterations = int(num_iterations)
        self._training_steps = training_steps
        self._evaluation_steps = evaluation_steps
        self._max_steps_per_episode = max_steps_per_episode
        self._clip_rewards = clip_rewards
        
        # Create environment and agent
        self.env = create_environment_fn()
        if explore_end_steps is None:
            explore_end_steps = training_steps - int(10e3)
        self._agent = create_agent_fn(self.env, explore_end_steps=explore_end_steps)

    def _initialize_episode(self, envs):
        """Initialize a new episode."""
        observations = []
        for env in envs:
            observations.append(env.reset())
        return np.stack(observations, 0)

    def __del__(self):
        """Clean up when the runner is destroyed."""
        if hasattr(self, '_agent'):
            del self._agent

@gin.configurable
class DataEfficientAtariRunner(Runner):
    """Runner for evaluating using a fixed number of episodes rather than steps."""

    def __init__(
        self,
        create_agent_fn,
        game_name=None,
        state_name=None,
        create_environment_fn=create_retro_environment,
        num_eval_episodes=100,
        max_noops=30,
        parallel_eval=True,
        num_eval_envs=100,
        num_train_envs=4,
        eval_one_to_one=True,
        record_episodes=False,
        recording_dir='episode_recordings',
    ):
        """Initialize the runner.
        
        Args:
            record_episodes: Whether to record episodes to HDF5
            recording_dir: Directory to save episode recordings
        """
        logging.info("game_name: {}".format(game_name))
        logging.info("state_name: {}".format(state_name))
        
        # Create a partial function with the game and state names
        create_environment_fn = functools.partial(create_environment_fn,
                                                game_name=game_name,
                                                state_name=state_name)
        
        # Initialize base class first
        super().__init__(create_agent_fn,
                         create_environment_fn=create_environment_fn)

        # Set derived class specific attributes
        self._num_eval_episodes = num_eval_episodes
        logging.info('Num evaluation episodes: %d', num_eval_episodes)
        self._evaluation_steps = None
        self.num_steps = 0
        self.total_steps = self._training_steps * self._num_iterations
        self.create_environment_fn = create_env_wrapper(create_environment_fn)

        self.max_noops = max_noops
        self.parallel_eval = parallel_eval
        self.num_eval_envs = num_eval_envs
        self.num_train_envs = num_train_envs
        self.eval_one_to_one = eval_one_to_one
        
        # Initialize episode recorder if enabled
        self.record_episodes = record_episodes
        if record_episodes:
            self.episode_recorder = EpisodeRecorder(recording_dir)
        else:
            self.episode_recorder = None

        # Initialize training environment
        self.train_envs = [self.env]  # Use the environment created by base class
        self.train_state = None
        
        # Initialize agent
        self._agent.reset_all(self._initialize_episode(self.train_envs))
        self._agent.cache_train_state()
        self.game_name = game_name.lower().replace('_', '').replace(' ', '')

    def _initialize_episode(self, envs):
        """Initialization for a new episode.

        Args:
            envs: Environments to initialize episodes for.

        Returns:
            action: int, the initial action chosen by the agent.
        """
        observations = []
        for env in envs:
            initial_observation = env.reset()
            if self.max_noops > 0:
                self._agent._rng, rng = jax.random.split(self._agent._rng)
                num_noops = jax.random.randint(rng, (), 0, self.max_noops)
                for _ in range(num_noops):
                    initial_observation, _, terminal, _ = env.step(0)
                    if terminal:
                        initial_observation = env.reset()
            observations.append(initial_observation)
        
        initial_observation = np.stack(observations, 0)
        return initial_observation

    def _run_one_phase(self,
                       envs,
                       steps,
                       max_episodes,
                       run_mode_str,
                       needs_reset=False,
                       one_to_one=False,
                       resume_state=None):
        """Runs the agent/environment loop until a desired number of steps.

    We terminate precisely when the desired number of steps has been reached,
    unlike some other implementations.

    Args:
      envs: environments to use in this phase.
      steps: int, how many steps to run in this phase (or None).
      max_episodes: int, maximum number of episodes to generate in this phase.
      run_mode_str: str, describes the run mode for this agent.
      needs_reset: bool, whether to reset all environments before starting.
      one_to_one: bool, whether to precisely match each episode in
        `max_episodes` to an environment in `envs`. True is faster but only
        works in some situations (e.g., evaluation).
      resume_state: bool, whether to have the agent resume its prior state for
        the current mode.

    Returns:
      Tuple containing the number of steps taken in this phase (int), the
      sum of
        returns (float), and the number of episodes performed (int).
    """
        step_count = 0
        num_episodes = 0
        sum_returns = 0.

        (episode_lengths, episode_returns, state, envs) = self._run_parallel(
            episodes=max_episodes,
            envs=envs,
            one_to_one=one_to_one,
            needs_reset=needs_reset,
            resume_state=resume_state,
            max_steps=steps,
        )

        for episode_length, episode_return in zip(episode_lengths,
                                                  episode_returns):
            if run_mode_str == 'train':
                # we use one extra frame at the starting
                self.num_steps += episode_length
            step_count += episode_length
            sum_returns += episode_return
            num_episodes += 1
            sys.stdout.flush()
        return step_count, sum_returns, num_episodes, state, envs

    def _run_parallel(self,
                      envs,
                      episodes=None,
                      max_steps=None,
                      one_to_one=False,
                      needs_reset=True,
                      resume_state=None):
        """Executes a full trajectory of the agent interacting with the environment."""
        # You can't ask for 200 episodes run one-to-one on 100 envs
        if one_to_one:
            assert episodes is None or episodes == len(envs)

        # Create envs
        live_envs = list(range(len(envs)))

        if needs_reset:
            new_obs = self._initialize_episode(envs)
            new_obses = np.zeros(
                (2, len(envs), *self._agent.observation_shape, 1))
            self._agent.reset_all(new_obs)

            rewards = np.zeros((len(envs),))
            terminals = np.zeros((len(envs),))
            episode_end = np.zeros((len(envs),))

            cum_rewards = []
            cum_lengths = []
        else:
            assert resume_state is not None
            (new_obses, rewards, terminals, episode_end, cum_rewards,
             cum_lengths) = (resume_state)

        total_steps = 0
        total_episodes = 0
        max_steps = np.inf if max_steps is None else max_steps
        step = 0

        # Keep interacting until we reach a terminal state.
        while True:
            b = 0
            step += 1
            episode_end.fill(0)
            total_steps += len(live_envs)
            
            # Log training progress every 1000 steps
            if total_steps % 1000 == 0:
                logging.info('Training progress: %d total steps, %d episodes completed', 
                           total_steps, total_episodes)
                if hasattr(self._agent, 'get_training_metrics'):
                    metrics = self._agent.get_training_metrics()
                    logging.info('Training metrics: %s', metrics)
            
            actions = self._agent.step()

            # The agent may be hanging on to the previous new_obs, so we don't
            # want to change it yet.
            # By alternating, we can make sure we don't end up logging
            # with an offset.
            new_obs = new_obses[step % 2]

            # don't want to do a for-loop since live envs may change
            while b < len(live_envs):
                env_id = live_envs[b]
                obs, reward, d, info = envs[env_id].step(actions[b])
                
                # Record the step if recording is enabled
                if self.record_episodes:
                    self.episode_recorder.record_step(
                        total_episodes,  # episode number
                        envs[env_id].cum_length,  # step number
                        obs,  # observation
                        actions[b],  # action
                        reward,  # reward
                        d,  # done
                        info  # info
                    )
                
                envs[env_id].cum_length += 1
                envs[env_id].cum_reward += reward
                new_obs[b] = obs
                rewards[b] = reward
                terminals[b] = d

                if (envs[env_id].game_over or
                        envs[env_id].cum_length == self._max_steps_per_episode):
                    total_episodes += 1
                    cum_rewards.append(envs[env_id].cum_reward)
                    cum_lengths.append(envs[env_id].cum_length)
                    
                    # Log episode completion
                    logging.info('Episode %d completed:', total_episodes)
                    logging.info('  Steps: %d', envs[env_id].cum_length)
                    logging.info('  Total reward: %.2f', envs[env_id].cum_reward)
                    logging.info('  Average reward per step: %.4f', 
                               envs[env_id].cum_reward / envs[env_id].cum_length)
                    
                    envs[env_id].cum_length = 0
                    envs[env_id].cum_reward = 0

                    human_norm_ret = normalize_score(cum_rewards[-1],
                                                     self.game_name)

                    logging.info(
                        'steps executed: {:>8}, '.format(total_steps) +
                        'num episodes: {:>8}, '.format(len(cum_rewards)) +
                        'episode length: {:>8}, '.format(cum_lengths[-1]) +
                        'return: {:>8}, '.format(cum_rewards[-1]) +
                        'normalized return: {:>8}'.format(
                            np.round(human_norm_ret, 3)))

                    if one_to_one:
                        new_obses = delete_ind_from_array(new_obses, b, axis=1)
                        new_obs = new_obses[step % 2]
                        actions = delete_ind_from_array(actions, b)
                        rewards = delete_ind_from_array(rewards, b)
                        terminals = delete_ind_from_array(terminals, b)
                        self._agent.delete_one(b)
                        del live_envs[b]
                        b -= 1  # live_envs[b] is now the next env, so go back one.
                    else:
                        episode_end[b] = 1
                        new_obs[b] = self._initialize_episode([envs[env_id]])
                        self._agent.reset_one(env_id=b)
                    # debug - start
                    if not self._agent.eval_mode:
                        self._agent.greedy_action = random.random(
                        ) < greedy_frac  #not self._agent.greedy_action
                        #logging.info("self._agent.greedy_action: {}".format(
                        #    self._agent.greedy_action))
                    # debug - end
                elif d:
                    self._agent.reset_one(env_id=b)
                    # debug - start
                    if not self._agent.eval_mode:
                        self._agent.greedy_action = random.random(
                        ) < greedy_frac  #not self._agent.greedy_action
                        #logging.info("self._agent.greedy_action: {}".format(
                        #    self._agent.greedy_action))
                    # debug - end

                b += 1

            if self._clip_rewards:
                # Perform reward clipping.
                rewards = np.clip(rewards, -1, 1)

            self._agent.log_transition(new_obs, actions, rewards, terminals,
                                       episode_end)

            if (not live_envs or
                (max_steps is not None and total_steps > max_steps) or
                (episodes is not None and total_episodes > episodes)):
                break

        state = (new_obses, rewards, terminals, episode_end, cum_rewards,
                 cum_lengths)
        return cum_lengths, cum_rewards, state, envs

    def _run_train_phase(self,):
        """Run training phase.

    Returns:
      num_episodes: int, The number of episodes run in this phase.
      average_reward: float, The average reward generated in this phase.
      average_steps_per_second: float, The average number of steps per
      second.
    """
        # Perform the training phase, during which the agent learns.
        self._agent.eval_mode = False
        # debug - start
        self._agent.greedy_action = random.random() < greedy_frac  #False
        # debug - end
        self._agent.restore_train_state()
        start_time = time.time()
        (
            number_steps,
            sum_returns,
            num_episodes,
            self.train_state,
            self.train_envs,
        ) = self._run_one_phase(
            self.train_envs,
            self._training_steps,
            max_episodes=None,
            run_mode_str='train',
            needs_reset=self.train_state is None,
            resume_state=self.train_state,
        )
        average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
        human_norm_ret = normalize_score(average_return, self.game_name)
        time_delta = time.time() - start_time
        average_steps_per_second = number_steps / time_delta
        logging.info('Average undiscounted return per training episode: %.2f',
                     average_return)
        logging.info('Average normalized return per training episode: %.2f',
                     human_norm_ret)
        logging.info('Average training steps per second: %.2f',
                     average_steps_per_second)
        self._agent.cache_train_state()
        return (
            num_episodes,
            average_return,
            average_steps_per_second,
            human_norm_ret,
        )

    def _run_eval_phase(self,):
        """Run evaluation phase.

    Returns:
        num_episodes: int, The number of episodes run in this phase.
        average_reward: float, The average reward generated in this phase.
    """
        # Perform the evaluation phase -- no learning.
        self._agent.eval_mode = True
        self._agent.greedy_action = True
        
        # For stable-retro, we can only have one environment at a time
        eval_env = self._create_environment()
        eval_envs = [eval_env]
        
        _, sum_returns, num_episodes, _, _ = self._run_one_phase(
            eval_envs,
            steps=None,
            max_episodes=self._num_eval_episodes,
            needs_reset=True,
            resume_state=None,
            one_to_one=True,
            run_mode_str='eval',
        )
        
        # Clean up the evaluation environment
        eval_env.close()
        
        average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
        logging.info(
            'Average undiscounted return per evaluation episode: %.2f',
            average_return,
        )
        human_norm_return = normalize_score(average_return, self.game_name)
        logging.info(
            'Average normalized return per evaluation episode: %.2f',
            human_norm_return,
        )
        return num_episodes, average_return, human_norm_return

    def _run_one_iteration(self,):
        """Runs one iteration of agent/environment interaction."""
        logging.info('Starting iteration %d', 0)
        (
            num_episodes_train,
            average_reward_train,
            average_steps_per_second,
            norm_score_train,
        ) = self._run_train_phase()
        
        # Enable evaluation phase
        num_episodes_eval, average_reward_eval, human_norm_eval = self._run_eval_phase()
        
        # Log training metrics
        logging.info('Training metrics:')
        logging.info('  Total frames: %d', self.num_steps)
        logging.info('  Training return: %.2f', average_reward_train)
        logging.info('  Training normalized score: %.2f', norm_score_train)
        logging.info('  Evaluation return: %.2f', average_reward_eval)
        logging.info('  Evaluation normalized score: %.2f', human_norm_eval)
        
        # Save metrics to a CSV file
        metrics_file = os.path.join(self.save_dir, 'training_metrics.csv')
        metrics = {
            'total_frames': self.num_steps,
            'train_return': average_reward_train,
            'train_norm_score': norm_score_train,
            'eval_return': average_reward_eval,
            'eval_norm_score': human_norm_eval
        }
        
        # Create or append to metrics file
        if not os.path.exists(metrics_file):
            with open(metrics_file, 'w') as f:
                f.write(','.join(metrics.keys()) + '\n')
        
        with open(metrics_file, 'a') as f:
            f.write(','.join(map(str, metrics.values())) + '\n')

    def run_experiment(self, eval_only=False, seed=None):
        """Runs a full experiment, spread over multiple iterations."""

        import orbax.checkpoint
        from flax.training import orbax_utils
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        directory = './single_save'
        import os
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_name = '{}/{}-{}.pth'.format(directory, self.game_name, seed)
        if not eval_only:
            logging.info('Beginning training...')
            self._run_one_iteration()
            # save jax model(s)
            ckpt = {
                'online_params': self._agent.online_params,
                'target_network_params': self._agent.target_network_params
            }
            save_args = orbax_utils.save_args_from_target(ckpt)
            orbax_checkpointer.save(file_name, ckpt, save_args=save_args)
        else:
            raw_restored = orbax_checkpointer.restore(file_name)
            self._agent.online_params = raw_restored['online_params']
            self._agent.target_network_params = raw_restored[
                'target_network_params']
            num_episodes_eval, average_reward_eval, human_norm_eval = (
                self._run_eval_phase())

    def __del__(self):
        """Clean up environments and episode recorder when the runner is destroyed."""
        super().__del__()
        for env in self.train_envs:
            env.close()
        if self.record_episodes:
            self.episode_recorder.close()


def delete_ind_from_array(array, ind, axis=0):
    start = tuple(([slice(None)] * axis) + [slice(0, ind)])
    end = tuple(([slice(None)] * axis) +
                [slice(ind + 1, array.shape[axis] + 1)])
    tensor = np.concatenate([array[start], array[end]], axis)
    return tensor

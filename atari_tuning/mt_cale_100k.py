from argparse import ArgumentParser
from functools import partial

import wandb

from amago.envs.builtin.ale_retro import AtariAMAGOWrapper, AtariGame
from amago.nets.cnn import NatureishCNN, IMPALAishCNN
from amago.cli_utils import *


def add_cli(parser):
    parser.add_argument("--games", nargs="+", default=None)
    parser.add_argument(
        "--cnn", type=str, choices=["nature", "impala"], default="impala"
    )
    parser.add_argument(
        "--action_space",
        type=str,
        choices=["discrete", "continuous", "multibinary"],
        default="continuous",
    )
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--memory_size", type=int, default=400)
    parser.add_argument("--memory_layers", type=int, default=3)
    parser.add_argument("--critics", type=int, default=6)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--grads_per_epoch", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--agent_type", type=str, choices=["multitask", "agent"], default="agent"
    )
    parser.add_argument("--actor_width", type=int, default=18)
    parser.add_argument("--no_log", action="store_true")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--buffer_dir", type=str, required=True)
    return parser


ATARI_100K_GAMES = [
    "Alien",
    "Amidar",
    "Assault",
    "Asterix",
    "BankHeist",
    "BattleZone",
    "Boxing",
    "Breakout",
    "ChopperCommand",
    "CrazyClimber",
    "DemonAttack",
    "Freeway",
    "Frostbite",
    "Gopher",
    "Hero",
    "Jamesbond",
    "Kangaroo",
    "Krull",
    "KungFuMaster",
    "MsPacman",
    "Pong",
    "PrivateEye",
    "Qbert",
    "RoadRunner",
    "Seaquest",
    "UpNDown",
]

DEBUG_SET = ["Boxing"]


def make_atari_100k_game(game_name):
    return AtariAMAGOWrapper(AtariGame(game=game_name))


ATARI_TIME_LIMIT = (30 * 60 * 60) // 4  # (30 minutes of game time)

if __name__ == "__main__":
    parser = ArgumentParser()
    add_cli(parser)
    args = parser.parse_args()

    RL_CONFIG = {
        "amago.nets.actor_critic.Actor.d_hidden": 300,
        "amago.nets.actor_critic.NCritics.d_hidden": 300,
        "amago.nets.actor_critic.NCriticsTwoHot.d_hidden": 300,
        "amago.nets.actor_critic.NCriticsTwoHot.output_bins": 64,
        "amago.nets.actor_critic.NCritics.num_critics": args.critics,
        "amago.nets.actor_critic.NCriticsTwoHot.num_critics": args.critics,
        "amago.nets.policy_dists.Discrete.clip_prob_low": 1e-4,
        "amago.nets.policy_dists.Discrete.clip_prob_high": 0.999,
        "amago.nets.policy_dists.TanhGaussian.clip_actions_on_log_prob": (
            -0.995,
            0.995,
        ),
        "amago.nets.policy_dists.TanhGaussian.std_activation": "softplus",
        "amago.nets.policy_dists.TanhGaussian.std_low": 1e-4,
        "amago.nets.policy_dists.TanhGaussian.std_high": 20.0,
    }
    ATARI_CONFIG = {
        "AtariGame.resolution": (84, 84),
        "AtariGame.grayscale": True,
        "AtariGame.channels_last": True,
        "AtariGame.sticky_action_prob": 0.0,
        "AtariGame.terminal_on_life_loss": True,
        "AtariGame.frame_skip": 4,
        "AtariGame.version": "v5",
        "AtariGame.action_space": args.action_space,
        "AtariGame.continuous_action_threshold": 0.5,
    }
    CONFIG = ATARI_CONFIG | RL_CONFIG
    traj_encoder_type = switch_traj_encoder(
        CONFIG,
        arch="transformer",
        memory_size=args.memory_size,
        layers=args.memory_layers,
    )
    cnn_encoder = switch_tstep_encoder(
        CONFIG,
        arch="cnn",
        cnn_type=NatureishCNN if args.cnn == "nature" else IMPALAishCNN,
        channels_first=False,
        drqv2_aug=True,
    )
    egreedy = switch_exploration(
        CONFIG,
        strategy="egreedy",
        eps_start=1.0,
        eps_end=0.01,
        steps_anneal=80_000,
    )
    agent_type = switch_agent(CONFIG, args.agent_type, reward_multiplier=1.0)
    use_config(CONFIG)

    # games = args.games or ATARI_100K_GAMES
    games = DEBUG_SET  # TODO Note: change this
    games *= args.actor_width
    env_funcs = [partial(make_atari_100k_game, game_name) for game_name in games]

    # inferring some hparams
    traj_file_length = args.seq_len * 5
    timesteps_per_epoch = traj_file_length
    # TODO: set long for debug
    epochs = 10 * 100_000 // (timesteps_per_epoch * args.actor_width)

    for trial in range(args.trials):
        # fmt: off
        experiment = amago.Experiment(
            run_name=f"{args.run_name}_trial_{trial}",
            max_seq_len=args.seq_len,
            traj_save_len=traj_file_length,
            tstep_encoder_type=cnn_encoder,
            traj_encoder_type=traj_encoder_type,
            agent_type=agent_type,
            make_train_env=env_funcs,
            make_val_env=env_funcs,
            parallel_actors=len(env_funcs),
            env_mode="sync",
            exploration_wrapper_type=egreedy,
            sample_actions=True,
            log_to_wandb=not args.no_log,
            wandb_project="amago-atari-100k",
            wandb_entity="ut-austin-rpl-general-team",
            dset_root=args.buffer_dir,
            dset_name=f"{args.run_name}_trial_{trial}",
            has_replay_buffer_rights=True,
            dset_max_size=float("inf"),
            save_trajs_as="npz",
            padded_sampling="both",
            dloader_workers=8,
            stagger_traj_file_lengths=False,
            epochs=1000,#epochs,
            start_learning_at_epoch=3,
            # SCHEDULE: this needs tuning with no budget,
            # then tuning to see what we can do in 100k
            #train_timesteps_per_epoch=timesteps_per_epoch,
            train_timesteps_per_epoch=1000,
            #train_timesteps_per_epoch=1000,
            #train_batches_per_epoch=args.grads_per_epoch,
            #val_interval=epochs // 4,
            val_interval=25,
            train_batches_per_epoch=1000,
            #val_timesteps_per_epoch=ATARI_TIME_LIMIT * max((3 // args.actor_width), 1),
            val_timesteps_per_epoch=5_000,
            # END SCHEDULE
            ckpt_interval=epochs // 4,
            always_save_latest=False,
            batch_size=args.batch_size,
            batches_per_update=1,
            learning_rate=1e-4,
            critic_loss_weight=10.0,
            lr_warmup_steps=500,
            grad_clip=2.0,
            l2_coeff=1e-3,
            local_time_optimizer=False,
        )
        # fmt: on

        experiment.start()
        experiment.learn()
        experiment.evaluate_test(
            env_funcs, timesteps=ATARI_TIME_LIMIT * 5, render=False
        )
        experiment.delete_buffer_from_disk()
        wandb.finish()

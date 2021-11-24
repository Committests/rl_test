import argparse
import sys


def get_ddpg_args_train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=200, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument(
        "--start_timesteps", default=1e4, type=int##TO HANGE FOR EXPL it was (1e4, 1e3 good) ---1e2 307 1e3 308 (3e3 good)
    )  # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=5e3, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=2.5e5, type=float)  # Max time steps to run environment for ----2.5e4 2.5e4good
    parser.add_argument("--save_models", action="store_true", default=True)  # Whether or not models are saved
    parser.add_argument("--expl_noise", default=0.3, type=float)  # Std of Gaussian exploration noise 0.1 TO CHANGE 0.2 (0.9 is good)
    parser.add_argument("--batch_size", default=32, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor 0.99 0.95 (97 is good)----------99   prov  206e 305 97, 306, 307 308:99 207 90, 95 from now on(95 good)
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument(
        "--policy_noise", default=0.2, type=float#was 0.2 (good 0.5)(0.7 is good)
    )  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--env_timesteps", default=500, type=int)  # Frequency of delayed policy updates
    parser.add_argument(
        "--replay_buffer_max_size", default=10000, type=int
    )  # Maximum number of steps to keep in the replay buffer
    parser.add_argument(
        "--log_file", default=None, type=str
    )  # Maximum number of steps to keep in the replay buffer

    return parser.parse_args()


def get_ddpg_args_test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=123, type=int)  # Inform the test what seed was used in training
    parser.add_argument("--experiment", default=2, type=int)

    return parser.parse_args()

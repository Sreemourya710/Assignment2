import argparse
from learning_algorithms import DQNTrainer
from utils import seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '-e', type=str, default='CartPole-v1')
    parser.add_argument('--rng_seed', '-rng', default=6369)
    parser.add_argument('--n_episode', '-ne', type=int, default=350, help='number of episode played (and trained) in total')
    parser.add_argument('--rm_cap', '-rmc', type=int, default=8192, help='replay-memory capacity')
    parser.add_argument('--batch_size', '-bs', type=int, default=128, help='number of random samples in one batch')
    parser.add_argument('--hidden_dim', '-hdim', type=int, default=128, help='hidden dimension of the q-net')
    parser.add_argument('--init_epsilon', '-init_eps', type=float, default=0.9, help='initial epsilon for epsilon-greedy policy')
    parser.add_argument('--min_epsilon', '-min_eps', type=float, default=0.05, help='minimum epsilon for epsilon-greedy policy')
    parser.add_argument('--epsilon_decay', '-decay', type=float, default=0.99, help='decay multiplier for epsilon')
    parser.add_argument('--gamma', '-gamma', type=float, default=0.99, help='discount gamma')
    parser.add_argument('--tau', '-tau', type=float, default=-0.005, help="q-net's weight in updating target-net by weighted-average")
    parser.add_argument('--lr', '-lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--exp_name', '-xn', type=str, default='my_exp', help='name of the experiment')
    args = parser.parse_args()
    params = vars(args)

    seed_everything(params['rng_seed'])

    trainer = DQNTrainer(params)
    trainer.run_training_loop()


if __name__ == '__main__':
    main()

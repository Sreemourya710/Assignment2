import argparse
from learning_algorithms import ACTrainer
from utils import seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '-e', type=str, default='LunarLander-v2')
    parser.add_argument('--rng_seed', '-rng', default=6369)
    parser.add_argument('--n_rollout', '-nr', type=int, default=100, help='number of rollouts played (and trained) in total')
    parser.add_argument('--n_trajectory_per_rollout', '-ntr', type=int, default=60, help='number of trajectories (episodes) per rollout to be gathered')
    parser.add_argument('--n_critic_iter', '-nci', type=int, default=1, help='number of times the critic updates the target value')
    parser.add_argument('--n_critic_epoch', '-nce', type=int, default=1, help='number of epochs for critic updates with one set of target value')
    parser.add_argument('--hidden_dim', '-hdim', type=int, default=128, help='hidden dimension of the policy-net')
    parser.add_argument('--actor_lr', '-alr', type=float, default=3e-3, help='learning rate of actor')
    parser.add_argument('--critic_lr', '-clr', type=float, default=3e-4, help='learning rate of critic')
    parser.add_argument('--exp_name', '-xn', type=str, default='my_exp', help='name of the experiment')
    args = parser.parse_args()
    params = vars(args)

    seed_everything(params['rng_seed'])

    trainer = ACTrainer(params)
    trainer.run_training_loop()


if __name__ == '__main__':
    main()

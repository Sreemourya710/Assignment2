import argparse
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', '-xn', type=str, default='my_exp', help='name of the experiment')
    args = parser.parse_args()
    params = vars(args)

    # Load pkl-file containing the learning (reward) history
    file_name = params['exp_name'] + '.pkl'
    with open(file_name, 'rb') as f:
        ro_reward = pickle.load(f)

    # Plot the data
    sns.lineplot(data=ro_reward, linestyle='--', label='tr0')
    plt.xlabel('rollout', fontsize=25, labelpad=-2)
    plt.ylabel('reward', fontsize=25)
    plt.title('Learning curve for CartPole', fontsize=30)
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()

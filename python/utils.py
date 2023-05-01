import random
import numpy as np
import torch.cuda


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# Util function to apply reward-discounting scheme on a list of instant-reward (from eq 8 of HW1)
def apply_discount(raw_reward, gamma=0.99):
    # TODO(DONE): Compute discounted_rtg_reward (as a list) from raw_reward
    # HINT: Reverse the input list, keep a running-average. Reverse again to get the correct order.
    discounted_rtg_reward = []
    discounted_reward = 0
    for r in reversed(raw_reward):
        # print(type(r), r)
        discounted_reward = r + discounted_reward * gamma
        discounted_rtg_reward.append(discounted_reward)
    discounted_rtg_reward = list(reversed(discounted_rtg_reward))

    # Normalization
    # print(type(discounted_rtg_reward))
    discounted_rtg_reward = np.array(discounted_rtg_reward)

    discounted_rtg_reward = discounted_rtg_reward - np.mean(discounted_rtg_reward) / (np.std(discounted_rtg_reward) + np.finfo(np.float32).eps)
    return torch.tensor(discounted_rtg_reward, dtype=torch.float32, device=get_device())

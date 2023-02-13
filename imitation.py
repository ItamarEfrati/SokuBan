import random

import matplotlib.pyplot as plt

import pickle

from data_modules.rl_dataset import Experience
from soko_pap import PushAndPullSokobanEnv


def play_step(env, state):
    while True:
        try:
            action = int(input("choose action"))
            new_state, reward, done, _ = env.step(action)
        except:
            print('again')
            continue
        break
    exp = Experience(state, action, reward, done, new_state)
    env.render()
    return exp, done


if __name__ == '__main__':
    env = PushAndPullSokobanEnv(dim_room=(7, 7), num_boxes=1, max_steps=500)

    experiences = []
    total_reward = 0
    for i in range(10):
        random.seed(2)
        is_done = False
        state = env.reset()
        env.render()
        while not is_done:
            exp, is_done = play_step(env, state)
            experiences.append(exp)

    with open('experiences2', 'wb') as f:
        pickle.dump(experiences, f)
    print(1)

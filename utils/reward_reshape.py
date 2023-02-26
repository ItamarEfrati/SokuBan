import numpy as np

from utils.multiprocess_env import SubprocVecEnv


def find_box_location(env_room_state):
    box_id = 4
    target_id = 2 if 2 in env_room_state else 5
    player_id = 5
    box_idx = np.argmax(env_room_state == box_id)
    target_idx = np.argmax(env_room_state == target_id)
    player_idx = np.argmax(env_room_state == player_id)
    return np.array(np.unravel_index(box_idx, env_room_state.shape)), \
           np.array(np.unravel_index(target_idx, env_room_state.shape)), \
           np.array(np.unravel_index(player_idx, env_room_state.shape))


def get_potential(env):
    env_room_state = env._get_room_states() if isinstance(env, SubprocVecEnv) else np.expand_dims(env.room_state, 0)
    batch_shape = env_room_state.shape[0]
    box_locs = np.zeros((batch_shape, 2))
    target_locs = np.zeros((batch_shape, 2))
    for i, s in enumerate(env_room_state):
        box_locs[i], target_locs[i], _ = find_box_location(s)

    return np.sum(np.abs(np.subtract(box_locs, target_locs)), axis=1)


def get_player_box_distance(env_room_state):
    box_loc, _, player_loc = find_box_location(env_room_state)
    return np.max(np.abs(np.subtract(box_loc, player_loc)))

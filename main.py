import os
import random

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from agents.actor_critic_agent import ActorCriticAgent
from agents.dqn_agents import DQNAgent
from models.dqn import D3QN
from env.soko_pap import PushAndPullSokobanEnv
from solvers.a2c_solver import A2CSolver
from solvers.dqn_solver import DQNSolver

IS_DEBUG = False
# CHECKPOINT = r'C:\Developments\SokuBan\results\6\checkpoints\epoch_73000.pt'
CHECKPOINT = None

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    latest_version = str(len(os.listdir('results'))) if not IS_DEBUG else 'debug'
    log_dir = os.path.join('results', latest_version)
    os.makedirs(os.path.join(log_dir, 'videos'), exist_ok=IS_DEBUG)
    os.makedirs(os.path.join(log_dir, 'checkpoints'), exist_ok=IS_DEBUG)

    torch.set_float32_matmul_precision('high')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # solver = DQNSolver(seed=2,
    #                    basic_net=D3QN,
    #                    device=device,
    #                    log_dir=log_dir,
    #                    log_every_n_epochs=2,
    #                    val_every=5,
    #                    num_episodes=200)

    solver = A2CSolver(seed=2,
                       device=device,
                       log_dir=log_dir)
    if CHECKPOINT is not None:
        solver.load_model(CHECKPOINT)
        solver.eval()
    try:
        solver.train()
    except KeyboardInterrupt:
        solver.save_statistics()
    finally:
        print('Closing envs')
        solver.train_env.close()

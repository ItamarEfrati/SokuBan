import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from agents.dqn_agents import DQNAgent
from models.dqn import D3QN
from env.soko_pap import PushAndPullSokobanEnv

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    train_env = PushAndPullSokobanEnv(dim_room=(7, 7), num_boxes=1, max_steps=500)
    val_env = PushAndPullSokobanEnv(dim_room=(7, 7), num_boxes=1, max_steps=500)
    agent = DQNAgent(train_env=train_env, val_env=val_env, seed=2, basic_net=D3QN)
    logger = TensorBoardLogger('.')
    trainer = Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=51,
        logger=logger
    )
    trainer.fit(agent)

import random
import matplotlib.pyplot as plt
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from models.dqn import DQNAgent
from soko_pap import PushAndPullSokobanEnv

# # Initialize the environment
# env = PushAndPullSokobanEnv(dim_room=(7, 7), num_boxes=1)
#
# # Initialize the seed to always get the same map.
# random.seed(2)
#
# # Reset the environment.
# env.reset()
#
# # Obtain the state.
# state = env.render("rgb_array")
#
# # Plot the state.
# plt.imshow(state)
# plt.show()
#
# random.seed(2)
# env.reset()
#
# # Obtain the state.
# state = env.render("rgb_array")
#
# # Plot the state.
# plt.imshow(state)
# plt.show()

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    train_env = PushAndPullSokobanEnv(dim_room=(7, 7), num_boxes=1, max_steps=500)
    val_env = PushAndPullSokobanEnv(dim_room=(7, 7), num_boxes=1, max_steps=500)
    agent = DQNAgent(train_env=train_env, val_env=val_env, seed=2)
    logger = TensorBoardLogger('lightning_logs')
    trainer = Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        max_epochs=51,
        logger=logger
    )
    trainer.fit(agent)



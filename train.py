from config import LABEL_COUNT
from models import Evaluations

print(LABEL_COUNT)
eval = Evaluations.get(Evaluations.id == 1)
print(eval.binary_base64())
print(eval.fen)
print(eval.eval)

####

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorchdatasets import EvalationDataSet, dataset

import time
from collections import OrderedDict

class EvaluationModel(pl.LightningModule):
  def __init__(self,learning_rate=1e-3,batch_size=1024,layer_count=10):
    super().__init__()
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    layers = []
    for i in range(layer_count-1):
      layers.append((f"linear-{i}", nn.Linear(808, 808)))
      layers.append((f"relu-{i}", nn.ReLU()))
    layers.append((f"linear-{layer_count-1}", nn.Linear(808, 1)))
    self.seq = nn.Sequential(OrderedDict(layers))

  def forward(self, x):
    return self.seq(x)

  def training_step(self, batch, batch_idx):
    x, y = batch['binary'], batch['eval']
    y_hat = self(x)
    loss = F.l1_loss(y_hat, y)
    self.log("train_loss", loss)
    return loss

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

  def train_dataloader(self):
    dataset = EvaluationDataset(count=LABEL_COUNT)
    return DataLoader(dataset, batch_size=self.batch_size, num_workers=2, pin_memory=True)

# Originally 4, 512, 1e-3
configs = [
           {"layer_count": 4, "batch_size": 256, "learning_rate": 1e-2, "max_epochs": 1},
           {"layer_count": 4, "batch_size": 256, "learning_rate": 1e-3, "max_epochs": 1},
           {"layer_count": 4, "batch_size": 256, "learning_rate": 1e-4, "max_epochs": 1},      
           {"layer_count": 6, "batch_size": 256, "learning_rate": 1e-2, "max_epochs": 1},
           {"layer_count": 6, "batch_size": 256, "learning_rate": 1e-3, "max_epochs": 1},
           {"layer_count": 6, "batch_size": 256, "learning_rate": 1e-4, "max_epochs": 1},
          #  {"layer_count": 6, "batch_size": 1024},
           ]

print(pl.__version__)
for config in configs:
  version_name = f'{int(time.time())}-batch_size-{config["batch_size"]}-layer_count-{config["layer_count"]}-learning_rate-{config["learning-rate"]}'
  logger = pl.loggers.TensorBoardLogger("lightning_logs", name="chessml", version=version_name)
  trainer = pl.Trainer(num_nodes=1,precision=16,max_epochs=config["max_epochs"],logger=logger)
  model = EvaluationModel(layer_count=config["layer_count"],batch_size=config["batch_size"],learning_rate=config["learning_rate"])
  # block commented out previously; appears to be for adaptive learning rate behavior, but the API has changed.
  #trainer.tune(model)
  #lr_finder = trainer.tuner.lr_find(model, min_lr=1e-6, max_lr=1e-3, num_training=25)
  #fig = lr_finder.plot(suggest=True)
  #fig.show()
  trainer.fit(model)
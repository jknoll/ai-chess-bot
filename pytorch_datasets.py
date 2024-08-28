# Defines the position evaluation dataset used by the PyTorch Lightning model for training. 

from torch.utils.data import IterableDataset
from random import randrange
from config import LABEL_COUNT
from database_models import Evaluations
import numpy as np

class SplitEvaluationDataset(IterableDataset):
    def __init__(self, start, end):
        super(SplitEvaluationDataset, self).__init__()
        self.start = start
        self.end = end

    def __iter__(self):
      return self

    def __next__(self):
      idx = randrange(self.start, self.end)
      return self[idx]
    
    def __len__(self):
      return self.end-self.start
    
    def __getitem__(self, idx):
      eval = Evaluations.get(Evaluations.id == idx+1)
      bin = np.frombuffer(eval.binary, dtype=np.uint8)
      bin = np.unpackbits(bin, axis=0).astype(np.single) 
      eval.eval = max(eval.eval, -15)
      eval.eval = min(eval.eval, 15)
      ev = np.array([eval.eval]).astype(np.single)

      # For testing mapping from FEN to bitboard
      return {'idx': idx, 'fen': eval.fen, 'binary':bin, 'eval':ev}    

      # Original return value
      # return {'binary':bin, 'eval':ev}    
      
# Docs for training, validation, test split implementation
# https://lightning.ai/docs/pytorch/stable/common/evaluation_basic.html

# Define splits
train_size = int(0.7 * LABEL_COUNT)
val_size = int(0.15 * LABEL_COUNT)
test_size = LABEL_COUNT - train_size - val_size
print(f"Training split size: {train_size}")
print(f"Validation split size: {val_size}")
print(f"Test split size: {test_size}")


# Create dataset splits
train_dataset = SplitEvaluationDataset(0, train_size)
val_dataset = SplitEvaluationDataset(train_size, train_size + val_size)
test_dataset = SplitEvaluationDataset(train_size + val_size, LABEL_COUNT)
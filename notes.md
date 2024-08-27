
**Observing/Logging Model While Training Quickly**

Discussion about how to log using on_after_backward without impairing training performance (here)[https://github.com/Lightning-AI/pytorch-lightning/issues/2077]

PyTorch Tensorboard tutorial (here)[https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html]. Doesn't cover how to capture gradients in a performant way.

Profiling tutorial: https://lightning.ai/docs/pytorch/stable/tuning/profiler_basic.html

It's clear that logging the histograms for tensorboard in EvaluationModel.on_after_backward() is causing execution to be much slower:
FIT Profiler Report

----------------------------------------------------------------------------------------------------------------------------------------------------------------
|  Action                                              	|  Mean duration (s)	|  Num calls      	|  Total time (s) 	|  Percentage %   	|
----------------------------------------------------------------------------------------------------------------------------------------------------------------
|  Total                                               	|  -              	|  703605         	|  5712.5         	|  100 %          	|
----------------------------------------------------------------------------------------------------------------------------------------------------------------
|  run_training_epoch                                  	|  570.49         	|  10             	|  5704.9         	|  99.868         	|
|  run_training_batch                                  	|  0.28607        	|  19540          	|  5589.7         	|  97.85          	|
|  [LightningModule]EvaluationModel.optimizer_step     	|  0.286          	|  19540          	|  5588.4         	|  97.828         	|
|  [Strategy]DDPStrategy.backward                      	|  0.28382        	|  19540          	|  5545.8         	|  97.082         	|
|  [LightningModule]EvaluationModel.on_after_backward  	|  0.282          	|  19540          	|  5510.4         	|  96.461         	|


**Adding Testing and Validation Steps**

**Maintenance: Upgrading wandb version, taking advice about tensor accuracy/speed tradeoff**

"You are using a CUDA device ('NVIDIA GeForce RTX 3090') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision"

**Unit Tests**

**Additional Data Collection**

After adding a training/validation/test split, validation_loss indicates that when training with 10M of the 37M original labeled examples, validation_loss starts to increase around epoch five, indicating overtraining. Training with the full 37M would help, but I suspect we might eventually hit another overfitting threshhold and that more data would be helpful.

Plan to add data:
1) Download a new lichess .pgn.zst archive. Complete
2) Download the original reference (July 2021) .pgn.zst archive.
3) Create a PGN => [(FEN, eval)...] parser which works on the lines containing the Stockfish eval. Complete
4) Pull the first eval-containing line from the original (July 2021) .pgn.zst. and parse it as in step 3, comparing to the first record(s) in the sqllite database. Complete: but the first eval-contining line in the July 2021 .pgn.zst is not the same as the first line in the database. Furthermore, all of the FEN I am generating with my parser is omitting the "en-passant eligible" FEN field. This is a design decision captured in this issue thread: https://github.com/niklasf/python-chess/issues/66

5) Add the FEN => bitboard binary representation to the parser from step 3. See https://chess.stackexchange.com/questions/29294/quickly-converting-board-to-bitboard-representation-using-python-chess-library

Open question: the evals in the lichess files sometimes indicate e.g. "mate in five", but the maximum evals in the sqlite database is 152.65

```
sqlite> select * from evaluations ORDER BY eval DESC limit 100;

45047|5R2/2K5/8/8/4k3/8/8/8 b - - 68 99||152.65
191443|8/8/8/4p3/1K5P/6P1/6N1/5k2 w - - 0 57||152.65
622203|8/8/6Kp/pk1Q2p1/1p4P1/7P/8/8 b - - 3 53||152.65
```


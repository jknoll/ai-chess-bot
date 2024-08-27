
**Observing/Logging Model While Training Quickly**

Discussion about how to log using on_after_backward without impairing training performance (here)[https://github.com/Lightning-AI/pytorch-lightning/issues/2077]

PyTorch Tensorboard tutorial (here)[https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html]. Doesn't cover how to capture gradients in a performant way.

Profiling tutorial: https://lightning.ai/docs/pytorch/stable/tuning/profiler_basic.html

**Adding Testing and Validation Steps**

**Maintenance: Upgrading wandb version, taking advice about tensor accuracy/speed tradeoff**

"You are using a CUDA device ('NVIDIA GeForce RTX 3090') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision"

**Unit Tests**

**Additional Data Collection**

After adding a training/validation/test split, validation_loss indicates that when training with 10M of the 37M original labeled examples, validation_loss starts to increase around epoch five, indicating overtraining. Training with the full 37M would help, but I suspect we might eventually hit another overfitting threshhold and that more data would be helpful.

Plan to add data:
1) Download another lichess .pgn.zst archive. Complete
2) Download the original reference (July 2021) .pgn.zst archive.
3) Create a PGN => [(FEN, eval)...] parser which works on the lines containing the Stockfish eval. Complete
4) Pull the first eval-containing line from the original (July 2021) .pgn.zst. and parse it as in step 3, comparing to the first record in the sqllite database.
5) Add the FEN => bitboard binary representation to the parser from step 3.
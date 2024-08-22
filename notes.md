
**Observing/Logging Model While Training Quickly**
Discussion about how to log using on_after_backward without impairing training performance (here)[https://github.com/Lightning-AI/pytorch-lightning/issues/2077]

PyTorch Tensorboard tutorial (here)[https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html]. Doesn't cover how to capture gradients in a performant way.

Profiling tutorial: https://lightning.ai/docs/pytorch/stable/tuning/profiler_basic.html

**Adding Testing and Validation Steps**
**Maintenance: Upgrading wandb version, taking advice about tensor accuracy/speed tradeoff**

"You are using a CUDA device ('NVIDIA GeForce RTX 3090') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision"

**Unit Tests**

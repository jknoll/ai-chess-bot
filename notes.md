
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

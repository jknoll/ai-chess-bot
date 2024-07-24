LABEL_COUNT = 37164639 # TODO: this should be read from the db, not hardcoded.
# LABEL_COUNT = 10000

# With logging, one full 37M Epoch ETA is 6h15m:
# Epoch 0:   0%|                                    | 61/72588 [00:18<6:15:56,  3.22it/s, v_num=13k0]
# Without, one full 37M Epoch ETA is 1h24m:
# Epoch 0:   0%|                                    | 188/72588 [00:13<1:24:40, 14.25it/s, v_num=0]
ENABLE_LOGGING = True
LOG_FREQUENCY = 500 # for the Trainer() constructor's log_every_n_steps; this defines a tradeoff between fine-grained logs and slower training.
# Currently unused
LOG_TENSORBOARD = True
LOG_WANDB = True

# Originally 4, 512, 1e-3
configs = [
           {"layer_count": 4, "batch_size": 512, "learning_rate": 1e-3, "max_epochs": 3}

#          {"layer_count": 4, "batch_size": 256, "learning_rate": 1e-3, "max_epochs": 1},
#          {"layer_count": 4, "batch_size": 256, "learning_rate": 1e-4, "max_epochs": 1},      
#          {"layer_count": 6, "batch_size": 256, "learning_rate": 1e-2, "max_epochs": 1},
#          {"layer_count": 6, "batch_size": 256, "learning_rate": 1e-3, "max_epochs": 1},
#          {"layer_count": 6, "batch_size": 256, "learning_rate": 1e-4, "max_epochs": 1},

           ]
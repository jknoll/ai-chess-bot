# LABEL_COUNT = 37164639 # TODO: this should be read from the db, not hardcoded.
LABEL_COUNT   = 10000000

# With logging, one full 37M Epoch ETA is 6h15m:
# Epoch 0:   0%|                                    | 61/72588 [00:18<6:15:56,  3.22it/s, v_num=13k0]
# Without, one full 37M Epoch ETA is 1h24m:
# Epoch 0:   0%|                                    | 188/72588 [00:13<1:24:40, 14.25it/s, v_num=0]
#
# From code isolation testing, it seems that the slowdown is from the on_after_backwards calls which log the histograms.

ENABLE_LOGGING = True
# LOG_FREQUENCY for the Trainer() constructor's log_every_n_steps; this defines a tradeoff between fine-grained logs and slower training.
# It seems that a "step" in every_n_steps means running one batch.
LOG_FREQUENCY = 500 
configs = [
          {"layer_count": 4, "batch_size": 512, "learning_rate": 1e-3, "max_epochs": 10}

#          {"layer_count": 4, "batch_size": 512, "learning_rate": 1e-2, "max_epochs": 10},
#          {"layer_count": 4, "batch_size": 512, "learning_rate": 1e-3, "max_epochs": 10},
#          {"layer_count": 4, "batch_size": 512, "learning_rate": 1e-4, "max_epochs": 10},      
#          {"layer_count": 6, "batch_size": 512, "learning_rate": 1e-2, "max_epochs": 10},
#          {"layer_count": 6, "batch_size": 512, "learning_rate": 1e-3, "max_epochs": 10},
#          {"layer_count": 6, "batch_size": 512, "learning_rate": 1e-4, "max_epochs": 10},
#          {"layer_count": 8, "batch_size": 512, "learning_rate": 1e-2, "max_epochs": 10},
#          {"layer_count": 8, "batch_size": 512, "learning_rate": 1e-3, "max_epochs": 10},
#          {"layer_count": 8, "batch_size": 512, "learning_rate": 1e-4, "max_epochs": 10},        
]
# LABEL_COUNT = 37164639 # TODO: this should be read from the .db file, not hardcoded.
LABEL_COUNT = 10000

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
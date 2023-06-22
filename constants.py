from models.client import cnn_2layers, cnn_3layers
from models.ResNet20 import resnet20

DEVICE = "cuda" # "cuda" or "cpu"

NUM_CLASSES = 100

BATCH_SIZE = 128     # Higher batch sizes allows for larger learning rates. An empirical heuristic suggests that, when changing
                     # the batch size, learning rate should change by the same factor to have comparable results

LR = 1e-3            # The initial Learning Rate
MOMENTUM = 0.9       # Hyperparameter for SGD, keep this at 0.9 when using SGD
WEIGHT_DECAY = 1e-3  # Regularization, you can keep this at the default

NUM_EPOCHS = 160     # Total number of training epochs (iterations over dataset)
STEP_SIZE = 10       # How many epochs before decreasing learning rate (if using a step-down policy)
GAMMA = 0.1          # Multiplicative factor for learning rate step-down

LOG_FREQUENCY = 1000
SEED = 21032000

# ====================================

MODELS_BALANCED = {
  "models": [{"model_type": "2_layer_CNN", "params": {"n1": 128, "n2": 256, "dropout_rate": 0.2}, "train_params": {"optimizer": "Adam", "lr": 1e-3}},
               {"model_type": "2_layer_CNN", "params": {"n1": 128, "n2": 384, "dropout_rate": 0.2}, "train_params": {"optimizer": "Adam", "lr": 1e-3}},
               {"model_type": "2_layer_CNN", "params": {"n1": 128, 'n2': 512, "dropout_rate": 0.2}, "train_params": {"optimizer": "Adam", "lr": 1e-3}},
               {"model_type": "2_layer_CNN", "params": {"n1": 256, "n2": 256, "dropout_rate": 0.3}, "train_params": {"optimizer": "Adam", "lr": 1e-3}},
               {"model_type": "2_layer_CNN", "params": {"n1": 256, "n2": 512, "dropout_rate": 0.4}, "train_params": {"optimizer": "Adam", "lr": 1e-3}},
               {"model_type": "3_layer_CNN", "params": {"n1": 64, "n2": 128, "n3": 256, "dropout_rate": 0.2}, "train_params": {"optimizer": "Adam", "lr": 1e-3}},
               {"model_type": "3_layer_CNN", "params": {"n1": 64, "n2": 128, "n3": 192, "dropout_rate": 0.2}, "train_params": {"optimizer": "Adam", "lr": 1e-3}},
               {"model_type": "3_layer_CNN", "params": {"n1": 128, "n2": 192, "n3": 256, "dropout_rate": 0.2}, "train_params": {"optimizer": "Adam", "lr": 1e-3}},
               {"model_type": "3_layer_CNN", "params": {"n1": 128, "n2": 128, "n3": 128, "dropout_rate": 0.3}, "train_params": {"optimizer": "Adam", "lr": 1e-3}},
               {"model_type": "ResNet20", "params": {}, "train_params": {"optimizer": "Adam", "lr": 1e-3}}
            ],
  "pre_train_params": {"min_delta": 0.005, "patience": 3,
                    "batch_size": 128, "epochs": 30, "is_shuffle": True, 
                    "verbose": 1},
  "model_saved_names" : ["CNN_128__256", "CNN_128_384", "CNN_128_512",
                          "CNN_256_256", "CNN_256_512", "CNN_64_128_256",
                          "CNN_64_128_192", "CNN_128_192_256", "CNN_128_128_128",
                          "ResNet20"],
  "N_agents": 10,
  "N_samples_per_class": 3,
  "N_subset": 5000, 
  "private_classes": [0,2,20,63,71,82],
  "public_classes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
  "N_rounds": 30,
  "N_logits_matching_round": 1, 
  "N_private_training_round": 4,
  "private_training_batchsize" : 5, 
  "logits_matching_batchsize": 256,
  "result_save_dir": "./result_CIFAR_balanced/"
}

MODELS_UNBALANCED = {
  "models": [{"model_type": "2_layer_CNN", "params": {"n1": 128, "n2": 256, "dropout_rate": 0.2}, "train_params": {"optimizer": "Adam", "lr": 1e-3}},
               {"model_type": "2_layer_CNN", "params": {"n1": 128, "n2": 384, "dropout_rate": 0.2}, "train_params": {"optimizer": "Adam", "lr": 1e-3}},
               {"model_type": "2_layer_CNN", "params": {"n1": 128, 'n2': 512, "dropout_rate": 0.2}, "train_params": {"optimizer": "Adam", "lr": 1e-3}},
               {"model_type": "2_layer_CNN", "params": {"n1": 256, "n2": 256, "dropout_rate": 0.3}, "train_params": {"optimizer": "Adam", "lr": 1e-3}},
               {"model_type": "2_layer_CNN", "params": {"n1": 256, "n2": 512, "dropout_rate": 0.4}, "train_params": {"optimizer": "Adam", "lr": 1e-3}},
               {"model_type": "3_layer_CNN", "params": {"n1": 64, "n2": 128, "n3": 256, "dropout_rate": 0.2}, "train_params": {"optimizer": "Adam", "lr": 1e-3}},
               {"model_type": "3_layer_CNN", "params": {"n1": 64, "n2": 128, "n3": 192, "dropout_rate": 0.2}, "train_params": {"optimizer": "Adam", "lr": 1e-3}},
               {"model_type": "3_layer_CNN", "params": {"n1": 128, "n2": 192, "n3": 256, "dropout_rate": 0.2}, "train_params": {"optimizer": "Adam", "lr": 1e-3}},
               {"model_type": "3_layer_CNN", "params": {"n1": 128, "n2": 128, "n3": 128, "dropout_rate": 0.3}, "train_params": {"optimizer": "Adam", "lr": 1e-3}},
               {"model_type": "ResNet20", "params": {}, "train_params": {"optimizer": "Adam", "lr": 1e-3}}
            ],
    "model_saved_names" : ["CNN_128__256", "CNN_128_384", "CNN_128_512",
                          "CNN_256_256", "CNN_256_512", "CNN_64_128_256",
                          "CNN_64_128_192", "CNN_128_192_256", "CNN_128_128_128",
                          "ResNet20"],
    "N_agents": 10,
    "N_samples_per_class": 20, # not 3
    "N_subset": 5000, 
    "private_classes": [0,1,7,9,12,18], # Super classes
    "public_classes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "N_rounds": 30,
    "N_logits_matching_round": 1, 
    "N_private_training_round": 10, # not 4
    "private_training_batchsize" : 10, # not 5
    "logits_matching_batchsize": 128, # not 256
    "result_save_dir": "./result_CIFAR_imbalanced/"
}

NETWORKS = {"2_layer_CNN": cnn_2layers, "3_layer_CNN": cnn_3layers, "ResNet20": resnet20}
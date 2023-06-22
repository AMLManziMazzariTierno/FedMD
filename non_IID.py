"""
Implementation of FedMD based on the TensorFlow implementation of diogenes0319/FedMD_clean
"""

"""
Implementation of FedMD based on the TensorFlow implementation of diogenes0319/FedMD_clean
"""

import argparse
import os
import argparse
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from constants import * 
import data.data_utils as data_utils
import data.CIFAR as CIFAR
import training as training
from FedMD import FedMD

from PIL import Image
from tqdm import tqdm
import wandb
from utils import *
from EarlyStop import *

def main():
    args = parse_args()    
    
    wandb_api_key = args.wandb
    os.environ["WANDB_API_KEY"] = wandb_api_key
    os.environ["WANDB_MODE"] = "online" if args.wandb else "offline"
    checkpoint_path = 'ckpt'
    paths = [checkpoint_path, f"{checkpoint_path}/ub"]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

    run_id = args.run_id
    restore_path = args.restore_path

    model_config = MODELS_UNBALANCED["models"]
    model_saved_names = MODELS_UNBALANCED["model_saved_names"]
    public_classes = MODELS_UNBALANCED["public_classes"]
    private_classes = MODELS_UNBALANCED["private_classes"]
    n_classes = len(public_classes) + len(private_classes)

    num_agents = MODELS_UNBALANCED["N_agents"]
    num_samples_per_class = MODELS_UNBALANCED["N_samples_per_class"]

    num_rounds = MODELS_UNBALANCED["N_rounds"]
    num_subset = MODELS_UNBALANCED["N_subset"]
    num_private_training_round = MODELS_UNBALANCED["N_private_training_round"]
    private_training_batch_size = MODELS_UNBALANCED["private_training_batchsize"]
    num_logits_matching_round = MODELS_UNBALANCED["N_logits_matching_round"]
    logits_matching_batch_size = MODELS_UNBALANCED["logits_matching_batchsize"]


    result_save_dir = MODELS_UNBALANCED["result_save_dir"]

    # Load CIFAR-10 and CIFAR-100 datasets
    print ("Loading CIFAR-10 and CIFAR-100 datasets...")

    train_cifar10, test_cifar10   = CIFAR.load_CIFAR10() # train_cifar10 = public_dataset
    train_cifar100, test_cifar100 = CIFAR.load_CIFAR100()
    train_coarse_cifar100, test_coarse_cifar100 = CIFAR.load_CIFAR100(granularity='coarse')
    
    print ("Dataset loaded.")
    
    # Create a list called relations to store the relationships between superclasses and subclasses in CIFAR-100
    relations = [set() for _ in range(20)]
    for i, y_fine in enumerate(train_cifar100.targets):
        relations[train_coarse_cifar100.targets[i]].add(y_fine)
    for i in range(len(relations)):
        relations[i] = list(relations[i])

    # Generate a partition of subclasses for each agent based on private_classes
    subclasses_per_agent = [[relations[j][i%5] for j in private_classes] for i in range(num_agents)]
    print("Subclasses partition per agent:")
    print(subclasses_per_agent)

    print ("Generating class subsets")
    # Create class subsets of the CIFAR-100 test dataset (test_cifar100)
    # containing only the private superclasses defined in private_classes
    # Map the subclasses to the range [10, 16) to ensure consistency with the number of public classes

    # Create a subset of dataset containing only the private super-classes to use
    test_cifar100.targets = test_coarse_cifar100.targets
    private_test_dataset = data_utils.generate_class_subset(test_cifar100, classes=private_classes)

    for index in range(len(private_classes)-1, -1, -1):
        cls_ = private_classes[index]
        private_test_dataset.targets[private_test_dataset.targets == cls_] = index + len(public_classes)
    #mod_private_classes = torch.arange(len(private_classes)) + len(public_classes)

    # Split the train dataset among the agent, with N_samples_per_class images per class per agent
    print (f"Splitting private dataset among {num_agents} agents, with {num_samples_per_class} images per class and agent")
    private_data, total_private_data = data_utils.split_dataset_imbalanced(train_cifar100, train_coarse_cifar100.targets, num_agents, num_samples_per_class, classes_per_agent=subclasses_per_agent, seed=SEED)
    for index in range(len(private_classes)-1, -1, -1):
        cls_ = private_classes[index]
        total_private_data.targets[total_private_data.targets == cls_] = index + len(public_classes)
        for i in range(num_agents):
            private_data[i].targets[private_data[i].targets == cls_] = index + len(public_classes)

    run, job_id, resumed = init_wandb(run_id=run_id, config=MODELS_UNBALANCED)

    # Create agent models
    agents = []
    for i, item in enumerate(model_config):
        model_name = item["model_type"]
        model_params = item["params"]
        train_params = item["train_params"]
        tmp = NETWORKS[model_name](n_classes=n_classes, input_shape=(3,32,32), **model_params)
        print("model {0} : {1}".format(i, model_saved_names[i]))
        agents.append({"model": tmp, "train_params": train_params})
        
        del model_name, model_params, tmp

    # Perform initial transfer learning training on the public dataset
    # For each model, check if a checkpoint exists and upload it if available; otherwise, perform training

    for i, agent in enumerate(agents):
        loaded = load_checkpoint(f"{checkpoint_path}/{model_saved_names[i]}_initial_pub.pt", agent["model"], restore_path)
        if not loaded:
            optimizer = training.load_optimizer(agent["model"], agent["train_params"])
            loss = nn.CrossEntropyLoss()
            print(f"Training {model_saved_names[i]}...")
            accuracies = training.train(network=agent["model"], 
                dataset=train_cifar10, 
                test_dataset=test_cifar10, 
                loss_fn=loss, 
                optimizer=optimizer, 
                batch_size=128, 
                num_epochs=20, 
                returnAcc=True
            )
            wandb.run.summary[f"{model_saved_names[i]}_initial_pub_test_acc"] = accuracies[-1]["test_accuracy"]
            
            torch.save(agent["model"].state_dict(), f'{checkpoint_path}/{model_saved_names[i]}_initial_pub.pt')
            wandb.save(f'{checkpoint_path}/{model_saved_names[i]}_initial_pub.pt')
        else:
            test_acc = training.test(network=agent["model"], test_dataset=test_cifar10, batch_size=128)
            wandb.run.summary[f"{model_saved_names[i]}_initial_pub_test_acc"] = test_acc

    fedmd = FedMD(agents, model_saved_names,
        public_dataset=train_cifar10, 
        private_data=private_data, 
        total_private_data=total_private_data, 
        private_test_data=private_test_dataset,
        num_rounds=num_rounds,
        num_subset=num_subset,
        num_logits_matching_round=num_logits_matching_round,
        logits_matching_batch_size=logits_matching_batch_size,
        num_private_training_round=num_private_training_round,
        private_training_batch_size=private_training_batch_size,
        restore_path=restore_path)
    
    collaboration = fedmd.collaborative_training()

    wandb.finish()

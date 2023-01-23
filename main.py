import argparse
import os
import logging
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import torchvision

from constants import * 
from client import cnn_2layers, cnn_3layers
from resnet20 import resnet20
import CIFAR
import train
from FedMD import FedMD
from wandb_utils import *

from PIL import Image
from tqdm import tqdm
import wandb

CANDIDATE_MODELS = {"2_layer_CNN": cnn_2layers, 
                    "3_layer_CNN": cnn_3layers,
                    "ResNet20": resnet20} 


def main():

    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} wandb_api_key")
        exit()
    
    wandb_api_key = sys.argv[1]
    os.environ["WANDB_API_KEY"] = wandb_api_key
    os.environ["WANDB_MODE"] = "online"
    ckpt_path = 'ckpt'
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    run_id = None
    if len(sys.argv) == 3:
        run_id = sys.argv[2]

    model_config = CONF_MODELS["models"]
    pre_train_params = CONF_MODELS["pre_train_params"]
    model_saved_dir = CONF_MODELS["model_saved_dir"]
    model_saved_names = CONF_MODELS["model_saved_names"]
    is_early_stopping = CONF_MODELS["early_stopping"]
    public_classes = CONF_MODELS["public_classes"]
    private_classes = CONF_MODELS["private_classes"]
    n_classes = len(public_classes) + len(private_classes)

    N_agents = CONF_MODELS["N_agents"]
    N_samples_per_class = CONF_MODELS["N_samples_per_class"]

    N_rounds = CONF_MODELS["N_rounds"]
    N_alignment = CONF_MODELS["N_subset"]
    N_private_training_round = CONF_MODELS["N_private_training_round"]
    private_training_batchsize = CONF_MODELS["private_training_batchsize"]
    N_logits_matching_round = CONF_MODELS["N_logits_matching_round"]
    logits_matching_batchsize = CONF_MODELS["logits_matching_batchsize"]


    result_save_dir = CONF_MODELS["result_save_dir"]

    # This dataset has 100 classes containing 600 images each. There are 500 training images and 100 testing images per 
    # class. The 100 classes in the CIFAR-100 are grouped into 20 superclasses. Each image comes with a "fine" label (the class to which it belongs) and a 
    # "coarse" label (the superclass to which it belongs).
    # Define transforms for training phase

    # random crop, random horizontal flip, per-pixel normalization 

    train_cifar10, test_cifar10   = CIFAR.load_CIFAR10() # train_cifar10 = public_dataset
    train_cifar100, test_cifar100 = CIFAR.load_CIFAR100()

    print ("=== Generating class subsets ===")

    private_train_dataset = CIFAR.generate_class_subset(train_cifar100, private_classes)
    private_test_dataset  = CIFAR.generate_class_subset(test_cifar100,  private_classes)

    for index, cls_ in enumerate(private_classes):        
        private_train_dataset.targets[private_train_dataset.targets == cls_] = index + len(public_classes)
        private_test_dataset.targets[private_test_dataset.targets == cls_] = index + len(public_classes)
    del index, cls_
    mod_private_classes = torch.arange(len(private_classes)) + len(public_classes)
    print (f"=== Splitting private dataset for the {N_agents} agents ===")

    private_data, total_private_data = CIFAR.split_dataset(private_train_dataset, N_agents, N_samples_per_class, classes_in_use=mod_private_classes)

    private_test_dataset = CIFAR.generate_class_subset(private_test_dataset, mod_private_classes)

    run, job_id, resumed = init_wandb(run_id=run_id)

    # X_train_CIFAR10 = train_cifar10.data
    # X_test_CIFAR10 = test_cifar10.data

    # y_train_CIFAR10 = train_cifar10.targets
    # y_test_CIFAR10 = test_cifar10.targets

    #X_train_CIFAR10, X_test_CIFAR10, y_train_CIFAR10, y_test_CIFAR10 = train_test_split(train_cifar10, test_cifar10, test_size=0.33, random_state=42)

    # public_dataset = {"X": X_train_CIFAR10, "y": y_train_CIFAR10}

    # private_data, total_private_data = CIFAR.split_dataset(train_cifar100, N_parties, N_samples_per_class)
    # mod_private_classes = np.arange(len(private_classes)) + len(public_classes)    
    # private_test_data = CIFAR.generate_class_subset(test_cifar100, mod_private_classes)
    
    # pre_models_dir = "./pretrained_CIFAR10/"

    agents = []
    for i, item in enumerate(model_config):
        model_name = item["model_type"]
        model_params = item["params"]
        tmp = CANDIDATE_MODELS[model_name](n_classes=n_classes, input_shape=(3,32,32), **model_params)
        print("model {0} : {1}".format(i, model_saved_names[i]))
        print("tmp = " + tmp)
        agents.append(tmp)
    
        del model_name, model_params, tmp
    #END FOR LOOP

    # pre_train_result = train_models(agents,train_cifar10,test_cifar10,num_epochs=NUM_EPOCHS,save_dir = pre_models_dir, save_names = model_saved_names)

    # In fedmd, each model is first trained on the public dataset, i.e., mnist dataset.
    # To save the experiment time, we provide pre_trained local models in the the "./pretrained_MNIST/" folder
    # If someone wants to conduct experiments from scratch, please use the above commented code to obtain the pretrained model

    # for i, item in enumerate(model_config):
    #     model_name = item["model_type"]
    #     model_params = item["params"]
    #     tmp = CANDIDATE_MODELS[model_name](n_classes=n_classes, **model_params)
    #     print("model {0} : {1}".format(i, model_saved_names[i]))
    #     print(tmp)
    #     tmp.load_state_dict(torch.load(os.path.join(pre_models_dir, "{}.h5".format(model_saved_names[i]))))
    #     agents.append(tmp)

    #     del model_name, model_params, tmp

    for i, agent in enumerate(agents):
        loaded = load_checkpoint(f"{ckpt_path}/{model_saved_names[i]}_initial_pub.pt", agents[i])
        if not loaded:
            optimizer = optim.Adam(agent.parameters(), lr = LR)
            loss = nn.CrossEntropyLoss()
            print(f"===== TRAINING {model_saved_names[i]} =====")
            accuracies = model_trainers.train_model(agent, train_cifar10, test_dataset=test_cifar10, loss_fn=loss, optimizer=optimizer, batch_size=128, num_epochs=20, returnAcc=True)
            best_test_acc = max(accuracies, key=lambda x: x["test_accuracy"])["test_accuracy"]
            wandb.run.summary[f"{model_saved_names[i]}_initial_pub_test_acc"] = best_test_acc
            
            torch.save(agent.state_dict(), f'{ckpt_path}/{model_saved_names[i]}_initial_pub.pt')
            wandb.save(f'{ckpt_path}/{model_saved_names[i]}_initial_pub.pt')
            #wandb.log({f"{model_saved_names[i]}_initial_test_acc": best_test_acc}, step=0)


    fedmd = FedMD(agents, model_saved_names,
                       public_dataset=train_cifar10,
                       private_data=private_data,
                       model_saved_name=model_saved_names,
                       total_private_data=total_private_data,
                       private_test_data=private_test_data,    
                       N_rounds=N_rounds,
                       N_subset=N_subset, # N_subset = N_alignment
                       N_logits_matching_round=N_logits_matching_round,
                       logits_matching_batchsize=logits_matching_batchsize,
                       N_private_training_round=N_private_training_round,
                       private_training_batchsize=private_training_batchsize)


    # initialization_result = fedmd.init_result

    collaboration_performance = fedmd.collaborative_training()
    wandb.finish()
# end main

if __name__ == '__main__':
    main()
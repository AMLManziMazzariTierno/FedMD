import wandb
import torch
import os
from constants import MODELS_BALANCED
import sys
import argparse

def init_wandb(run_id=None, config=MODELS_BALANCED):
    group_name = "fedmd"

    configuration = config
    agents = ""
    for agent in configuration["models"]:
        agents += agent["model_type"][0]
    job_name = f"M{configuration['N_agents']}_N{configuration['N_rounds']}_S{config['N_subset']}_A{agents}"

    run = wandb.init(
                id = run_id,
                # Set entity to specify your username or team name
                entity="samaml",
                # Set the project where this run will be logged
                project='fl_md',
                group=group_name,
                # Track hyperparameters and run metadata
                config=configuration,
                resume="allow")

    if os.environ["WANDB_MODE"] != "offline" and not wandb.run.resumed:
        random_number = wandb.run.name.split('-')[-1]
        wandb.run.name = job_name + '-' + random_number
        wandb.run.save()
        resumed = False
    if wandb.run.resumed:
        resumed = True

    return run, job_name, resumed


def load_checkpoint(path, model, restore_path = None):
    loaded = False
    if wandb.run.resumed or restore_path is not None:
        try:
            weights = wandb.restore(path, run_path=restore_path)
            model.load_state_dict(torch.load(weights.name))
            print(f"Loaded checkpoint {path}")
            loaded = True
        except ValueError:
            print(f"Checkpoint {path} does not exist")            
        except RuntimeError:
            print(f"Checkpoint {path} is corrupted")
            files = wandb.run.files()
            for file in files:
                if file.name == path:
                    file.delete()
            print("Deleted, please restart the run.")
    return loaded

def parse_args():
    parser = argparse.ArgumentParser(description='FedMD, a federated learning framework. \
    Participants are training collaboratively. ')
    parser.add_argument('-wandb', metavar='wandb',
                        help='the wandb API key.'
                        )
    parser.add_argument('-run_id', metavar='run_id',
                        help='the wandb run id to resume.'
                        )
    parser.add_argument('-restore_path', metavar='restore_path',
                        help='the wandb project path to restore files.'
                        )
    args = None
    if len(sys.argv) > 1:
        args = parser.parse_args(sys.argv[1:])
    return args

import wandb
import torch
import os
from constants import *
## -- WANDB --

os.environ["WANDB_MODE"] = "online"

def init_wandb(run_id=None):
    group_name = "fedmd"

    configuration = CONF_MODELS
    agents = ""
    for agent in configuration["models"]:
        agents += agent["model_type"][0]
    job_name = f"M{configuration['N_agents']}_N{configuration['N_rounds']}_S{CONF_MODELS['N_subset']}_lr{LR}_A{agents}"

    run = wandb.init(
                id = run_id,
                # Set entity to specify your username or team name
                entity="samaml",
                # Set the project where this run will be logged
                project='fl_fedmd',
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


def load_checkpoint(path, model):
    loaded = False
    if wandb.run.resumed:
        try:
            weights = wandb.restore(path)
            model.load_state_dict(torch.load(weights.name))
            print(f"===== SUCCESFULLY LOADED {path} FROM CHECKPOINT =====")
            loaded = True
        except ValueError:
            print(f"===== CHECKPOINT FOR {path} DOES NOT EXIST =====")            
        except RuntimeError:
            print(f"===== CHECKPOINT FOR {path} IS CORRUPTED =====")
            print("Deleting...")
            files = wandb.run.files()
            for file in files:
                if file.name == path:
                    file.delete()
            print("Deleted. Sorry for the inconveniences")
    return loaded
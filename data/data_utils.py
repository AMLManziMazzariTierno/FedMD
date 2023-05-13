import torch
from data.custom_subset import CustomSubset as Subset

def generate_class_subset(dataset, classes):
    """
    Generates a torch Subset including only the samples which label is contained in classes

    Parameters:
        dataset: torch Dataset
        classes: Array[Int] with the class labels 

    Returns: CustomSubset of selected images
    """
    dataset_classes = torch.tensor(dataset.targets)
    idxs = torch.cat([torch.nonzero(dataset_classes == i) for i in classes])
    return Subset(dataset, idxs)

def split_dataset(dataset, N_agents, N_samples_per_class, classes_in_use = None, seed = None):
    """
    Generates a random class balanced split of the dataset among N_agents,
    assigning N_samples_per_class samples per class per agent.

    Parameters:
        dataset: torch Dataset or CustomSubset to be splitted
        N_agents: Int of the number of splits to be done
        N_samples_per_class: Int of number of samples per each class to be assigned to each agent
        classes_in_use: Array[Int] with the class labels to be sampled from. If it's not specified
                        the whole set of labels will be used.
        seed: Seed to control the random function
    """

# The function first creates an empty list of indexes for each agent.
# It then selects a random set of samples for each class and assigns
# them to the agents. It does this by using the torch.multinomial()
# function, which returns a set of indices that correspond to the samples
# that are selected. These indices are then used to create a Subset object
# for each agent.

    if classes_in_use is None:
        classes_in_use = list(set(dataset.targets))
    if seed is not None:
        rand_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    # Labels for each sample in the dataset
    labels = torch.tensor(dataset.targets)

    # List of empty tensors, one for each agent
    private_idxs = [torch.tensor([], dtype=torch.long)]*N_agents

    # Empty tensor that will contain all the indices
    all_idxs = torch.tensor([], dtype=torch.long)

    # For each class
    for cls_ in classes_in_use:

        # Get the indices of all samples in that class
        idxs = torch.nonzero(labels == cls_).flatten()

        # From these, we sample (N_agents * N_samples_per_class) indices,
        # which represent a random set of selected samples for each class.
        samples = torch.multinomial(torch.ones(idxs.size()), N_agents * N_samples_per_class)
        
        # Concatenate the sampled indices to all_idxs
        all_idxs = torch.cat((all_idxs, idxs[samples]))

        for i in range(N_agents):
            # Get (N_samples_per_class) indices from the sampled indices for each class, using slicing.
            # These will be the indices that we assign to each agent.
            idx_agent = idxs[samples[i*N_samples_per_class : (i+1)*N_samples_per_class]]
            
            # The selected indices for that agent are concatenated to the corresponding
            # tensor in private_idxs
            private_idxs[i] = torch.cat((private_idxs[i], idx_agent))

    # If seed is not None, the function resets the random state
    # to what it was before the sampling was done.
    if seed is not None:
        torch.random.set_rng_state(rand_state)

    # We create a list of Subset objects, one for each agent.
    # So a Subset object corresponds to all the samples assigned to one of the agents.
    private_data = [Subset(dataset, private_idx) for private_idx in private_idxs]
    
    # a final Subset object that contains all the samples assigned to all agents
    all_private_data = Subset(dataset, all_idxs)
    
    return private_data, all_private_data

def split_dataset_imbalanced(dataset, super_classes, N_agents, N_samples_per_class, classes_per_agent, seed = None):
    """
    Generates a random class imbalanced split of the dataset among N_agents,
    assigning N_samples_per_class samples per class per agent, where each agent is assigned
    a specific set of classes_per_agent.
    It is worth noting that this function assumes that the dataset has non-overlapping classes.
    Otherwise, some samples might be assigned to more than one agent.

    Parameters:
        dataset: torch Dataset or CustomSubset to be splitted
        super_classes: Array of class labels to be assigned to the dataset
        N_agents: Int of the number of splits to be done
        N_samples_per_class: Int of number of samples per each class to be assigned to each agent
        classes_per_agent: Array[Array[Int]] with the class labels to be sampled from for each specific agent.
        seed: Seed to control the random function
    """
    if seed is not None:
        # Set the random generator state
        rand_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    # Extract the labels of the dataset
    labels = torch.tensor(dataset.targets)

    # Empty list of indices for each agent,
    # which will be used to create the subsets later
    private_idxs = [torch.tensor([], dtype=torch.long)]*N_agents

    # Empty tensor to store all the indices of the samples
    # assigned to each agent
    all_idxs = torch.tensor([], dtype=torch.long)

    # Loop over each agent
    for i, agent_classes in enumerate(classes_per_agent):
        # Loop over each class assigned to that agent
        for cls_ in agent_classes:

            idxs = torch.nonzero(labels == cls_).flatten()

            # Select (N_samples_per_class) random samples
            samples = torch.multinomial(torch.ones(idxs.size()), N_samples_per_class)
            idx_agent = idxs[samples]

            # Concatenate the indices of the selected samples
            # to private_idxs for the corresponding i-th agent
            private_idxs[i] = torch.cat((private_idxs[i], idx_agent))

        # Concatenate the indices of the assigned samples assigned into all_idxs
        all_idxs = torch.cat((all_idxs, private_idxs[i]))

    if seed is not None:
        # Set the random generator state back to its original state
        torch.random.set_rng_state(rand_state)

    dataset.targets = super_classes

    # Generate the subsets for each agent
    private_data = [Subset(dataset, private_idx) for private_idx in private_idxs]
    
    # Subset containing all samples assigned to all agents.
    all_private_data = Subset(dataset, all_idxs)
    
    return private_data, all_private_data

# In summary, the function generates a class-imbalanced split
# of a dataset by randomly assigning samples of each class to a
# specified number of agents, while ensuring that each agent is
# assigned a specific set of classes.


def stratified_sampling(dataset, size = 3000):
    """
    Generates a stratified-sampled subset using the dataset labels as categories
    
    Parameters:
        dataset: pytorch Dataset to be sample from
        size: Sample size 
    """
    import sklearn.model_selection
    idxs = sklearn.model_selection.train_test_split([i for i in range(len(dataset))], \
        train_size = size, stratify = dataset.targets)[0]
    return Subset(dataset, idxs)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torch.backends import cudnn
from constants import *
#from pytorchtools import EarlyStopping

import torchvision

def train_model(network, train_dataset, test_dataset, patience, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS, scheduler = None, returnAcc = False):
  # TODO: Implement Accuracy calculation for both validation and training. Also implement early stopping.
  # returned accuracy should be a list where accuracy = [training_accuracy, validation_accuracy]
  train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
  # By default, everything is loaded to cpu
  net = network.to(DEVICE) # this will bring the network to GPU if DEVICE is cuda
  
  #early_stopping = EarlyStopping(patience=patience, verbose=True)
  
  cudnn.benchmark # Calling this optimizes runtime

  current_step = 0
  # Start iterating over the epochs
  # training, validation history
  if returnAcc:
    accuracy = []
  for epoch in range(num_epochs):
    running_corrects = 0
    if scheduler is not None:
      print('Starting epoch {}/{} LR={}'.format(epoch+1, num_epochs, scheduler.get_lr()))
    else:
      print('Starting epoch {}/{}'.format(epoch+1, num_epochs))

    i = 0
    # Iterate over the dataset
    for images, labels in train_dataloader:
      # Bring data over the device of choice
      images = images.to(DEVICE)
      labels = labels.to(DEVICE)

      net.train() # Sets module in training mode

      # PyTorch, by default, accumulates gradients after each backward pass
      # We need to manually set the gradients to zero before starting a new iteration
      optimizer.zero_grad() # Zero-ing the gradients

      # Forward pass to the network
      outputs = net(images)

      # Compute loss based on output and ground truth
      loss = loss_fn(outputs, labels)

      # Log loss
      if current_step % LOG_FREQUENCY == 0:
        print('Step {}, Loss {}'.format(current_step, loss.item()))
      _, preds = torch.max(outputs.data, 1)
      # Update Corrects
      running_corrects += torch.sum(preds == labels.data).data.item()

      # Compute gradients for each layer and update weights
      loss.backward()  # backward pass: computes gradients
      optimizer.step() # update weights based on accumulated gradients
      if scheduler is not None:
        scheduler.step(epoch + i / len(train_dataloader))
      i += 1

      current_step += 1
    # end for dataset iteration

    # Step the scheduler
    # scheduler.step()
    if returnAcc:
      train_accuracy = running_corrects / len(dataset)
      test_accuracy = test_network(net, test_dataset, batch_size=BATCH_SIZE)
      accuracy.append([train_accuracy, test_accuracy])

  #end of epoch
  if returnAcc:
    return accuracy
  
def test_network(network, test_dataset, batch_size=BATCH_SIZE):
  test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
  net = network.to(DEVICE) # this will bring the network to GPU if DEVICE is cuda
  net.train(False) # Set Network to evaluation mode

  running_corrects = 0
  for images, labels in test_dataloader:
    images = images.to(DEVICE)
    labels = labels.to(DEVICE)

    # Forward Pass
    outputs = net(images)

    # Get predictions
    _, preds = torch.max(outputs.data, 1)

    # Update Corrects
    running_corrects += torch.sum(preds == labels.data).data.item()

  # Calculate Accuracy
  accuracy = running_corrects / float(len(test_dataset))
  return accuracy

def run_dataset(network, dataset, batch_size=BATCH_SIZE):
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
  net = network.to(DEVICE) # this will bring the network to GPU if DEVICE is cuda
  net.train(False) # Set Network to evaluation mode

  total_values = torch.tensor([])
  for images, labels in dataloader:
    images = images.to(DEVICE)
    labels = labels.to(DEVICE)

    # Forward Pass
    outputs = net(images)
    # Sample 1 -> [2.3, 4.1, 4.3, ..., ]
    # Sample .. ->[3.1, 1.3, 2.4, ..., ]

    # Get predictions
    total_values = torch.cat((total_values, outputs.data))
    # Update Corrects

  # Calculate Accuracy
  return 

def train_and_eval(model, train_loader, dev_loader, num_epochs,batch_size):


    model_trained=train_model(model,train_loader,epochs=num_epochs,batch_size=batch_size)

    # ********************* Evaluate for one epoch on training and validation set *********************
    val_metrics = test_network(model_trained, dev_loader, cuda=True)     # {'acc':acc, 'loss':loss}
    train_metrics = test_network(model_trained, train_loader, cuda=True)     # {'acc':acc, 'loss':loss}

    val_acc = val_metrics['acc']
    val_loss=val_metrics['loss']
    train_acc = train_metrics['acc']
    train_loss=train_metrics['loss']

    print('Val acc:',val_acc)
    print('Val loss:',val_loss)
    print('Train acc:',train_acc)
    print('Train loss:',train_loss)





    return model_trained, train_acc, train_loss, val_acc, val_loss


def train_models(models, train_loader, dev_loader,num_epochs, save_dir, save_names):

  resulting_val_acc = []
  record_result = []

  for n, model in enumerate(models):
        print("Training model ", n)
        model,train_acc,train_loss,val_acc,val_loss=train_and_eval(model,train_loader,dev_loader,num_epochs,batch_size=BATCH_SIZE)


        resulting_val_acc.append(val_acc)
        record_result.append({"train_acc": train_acc,
                              "val_acc": val_acc,
                              "train_loss": train_loss,
                              "val_loss": val_loss})

  print("pre-train accuracy:")
  print(resulting_val_acc)

  return record_result
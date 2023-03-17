import os
import torch
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from data import CTDataset
from utils import train, create_pred
from model import UNet


if __name__ == '__main__':
    # Setting writer to log results
    runs = [dir for dir in os.listdir('logs/') if os.path.isdir(f'logs/{dir}')]

    if not runs:
        new_run = 1

    else:
        new_run = max([int(el.split('_')[1]) for el in os.listdir('logs/')]) + 1

    writer = SummaryWriter(f'logs/run_{new_run}')

    ####################################################################################################################
    ################################################### First phase ####################################################
    ####################################################################################################################
    # Hyperparameters for first phase
    lr_1 = 1e-3
    batch_size_1 = 2
    num_epochs_1 = 30

    # Define transforms
    transform = transforms.Compose([transforms.Normalize(0, 1)])

    # Create dataset and split train and val loaders
    dataset = CTDataset('train', transform=transform)
    validation_split = .2

    # Creating data indices for training and validation splits
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating data samplers and loaders
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader_1 = DataLoader(dataset, batch_size=batch_size_1, sampler=train_sampler)
    val_loader_1 = DataLoader(dataset, batch_size=batch_size_1, sampler=valid_sampler)

    # Creating model
    model = UNet()

    # Train on initial labeled data
    train(model, train_loader_1, val_loader_1, num_epochs=num_epochs_1, lr=lr_1, logger=writer)

    ####################################################################################################################
    ################################################### Second phase ###################################################
    ####################################################################################################################
    # Dataset for unlabeled data
    unlabeled_dataset = CTDataset('unlabeled', transform=transform)

    # Create labels for unlabeled examples
    print("Creating labels for unlabeled data")
    create_pred(model, unlabeled_dataset)

    # New hyperparemeters for second phase
    lr_2 = 1e-3
    batch_size_2 = 2
    num_epochs_2 = 30

    # Create new dataloader with full dataset
    train_loader_2 = DataLoader(CTDataset('full'),
                                batch_size=batch_size_2,
                                shuffle=True)

    # Retrain on new data with new examples
    train(model, train_loader_2, num_epochs=num_epochs_2, lr=lr_2, logger=writer)

    ####################################################################################################################
    ################################################### Third phase ####################################################
    ####################################################################################################################
    # We recreate new labels for the original unlabeled dataset
    # Dataset for unlabeled data
    unlabeled_dataset = CTDataset('unlabeled', transform=transform)

    # Create labels for unlabeled examples
    print("Creating labels for unlabeled data")
    create_pred(model, unlabeled_dataset)

    # New hyperparemeters for second phase
    lr_3 = 1e-3
    batch_size_3 = 2
    num_epochs_3 = 30

    # Create new dataloader with full dataset
    train_loader_3 = DataLoader(CTDataset('full'),
                                batch_size=batch_size_3,
                                shuffle=True)

    # Retrain on new data with new examples
    train(model, train_loader_3, num_epochs=num_epochs_3, lr=lr_3, logger=writer)



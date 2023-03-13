from torch.utils.data import DataLoader
from data import CTDataset
from utils import train, create_pred
from model import UNet


if __name__ == '__main__':
    ####################################################################################################################
    ################################################### First phase ####################################################
    ####################################################################################################################
    # Hyperparameters for first phase
    lr_1 = 0.001
    batch_size_1 = 8
    num_epochs_1 = 5

    # Loader for labeled data
    train_loader_1 = DataLoader(CTDataset('train'),
                                batch_size=batch_size_1,
                                shuffle=True)

    # Creating model
    model = UNet()

    # Train on initial labeled data
    train(model, train_loader_1, num_epochs=num_epochs_1, lr=lr_1)

    ####################################################################################################################
    ################################################### Second phase ###################################################
    ####################################################################################################################
    # Dataset for unlabeled data
    unlabeled_dataset = CTDataset('unlabeled')

    # Create labels for unlabeled examples
    print("Creating labels for unlabeled data")
    create_pred(model, unlabeled_dataset)

    # New hyperparemeters for second phase
    lr_2 = 0.001
    batch_size_2 = 8
    num_epochs_2 = 5

    # Create new dataloader with full dataset
    train_loader_2 = DataLoader(CTDataset('full'),
                                batch_size=batch_size_2,
                                shuffle=True)

    # Retrain on new data with new examples
    train(model, train_loader_2, num_epochs_2, lr=lr_2)



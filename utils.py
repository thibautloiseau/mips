import torch
from tqdm import tqdm
import cv2
import numpy as np
from losses import GlobalLoss


def train(model, train_loader, num_epochs=10, lr=0.001):
    cuda = True if torch.cuda.is_available() else False
    print(f"Using cuda device: {cuda}")  # check if GPU is used

    # Tensor type (put everything on GPU if possible)
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # Loss function
    # To change -> permutation independent
    # We just take L1Loss to make it run at first
    # criterion = torch.nn.L1Loss()
    criterion = GlobalLoss()

    if cuda:
        model = model.cuda()
        criterion.cuda()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ----------
    #  Training
    # ----------
    for epoch in range(num_epochs):
        global_loss = 0.0

        for i, batch in tqdm(enumerate(train_loader)):
            slice = batch['slice'].type(Tensor)
            seg = batch['seg'].type(Tensor)

            # Remove stored gradients
            optimizer.zero_grad()

            # Generate output
            y_pred = model(slice)

            # Compute the corresponding loss
            loss = criterion(y_pred, seg)
            global_loss += loss.item()

            # Compute the gradient and perform one optimization step
            loss.backward()
            optimizer.step()

        print(f'Epoch: {epoch+1}, Loss: {global_loss}')

    return


def create_pred(model, unlabeled_dataset):
    """Create predictions for unlabeled data for co-training"""
    cuda = True if torch.cuda.is_available() else False
    print(f"Using cuda device: {cuda}")  # check if GPU is used

    # Tensor type (put everything on GPU if possible)
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    if cuda:
        model = model.cuda()

    with torch.no_grad():
        for sample in unlabeled_dataset:
            slice = sample['slice'].type(Tensor)[None, :, :, :]
            path_saving = sample['data_path'].split('\\')[-1]

            # Creating prediction for unlabeled data
            y_pred = torch.round(model(slice)).cpu().numpy().squeeze().reshape(512, 512, 1).astype(np.uint8)

            # Saving prediction
            cv2.imwrite(f'data/y_train/{path_saving}', y_pred)

    return

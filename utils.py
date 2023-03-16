import torch
from tqdm import tqdm
import cv2
import numpy as np
from losses import GlobalLoss
import sklearn.metrics as m


def train(model, train_loader, val_loader=None, num_epochs=10, lr=0.001, logger=None):
    cuda = True if torch.cuda.is_available() else False
    print(f"Using cuda device: {cuda}")  # check if GPU is used

    # Tensor type (put everything on GPU if possible)
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # Loss function
    criterion = GlobalLoss()

    if cuda:
        model = model.cuda()
        criterion.cuda()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=10)

    # ----------
    #  Training
    # ----------
    for epoch in range(num_epochs):
        train_loss, eval_loss = [], []
        train_score, eval_score = [], []

        # Training
        for i, batch in tqdm(enumerate(train_loader)):
            slice = batch['slice'].type(Tensor)
            seg = batch['seg'].type(Tensor)

            # Remove stored gradients
            optimizer.zero_grad()

            # Generate output
            y_pred = model(slice)

            # Compute the corresponding loss
            loss = criterion(y_pred, seg)
            train_loss.append(loss.item())

            # Compute the gradient and perform one optimization step
            loss.backward()
            optimizer.step()

            # Compute metric
            train_score.append(rand_score(y_pred, seg))

        if val_loader is not None:
            # Evaluation
            with torch.no_grad():
                for i, batch in tqdm(enumerate(val_loader)):
                    slice = batch['slice'].type(Tensor)
                    seg = batch['seg'].type(Tensor)

                    # Generate output
                    y_pred = model(slice)

                    # Compute the corresponding loss
                    loss = criterion(y_pred, seg)
                    eval_loss.append(loss.item())

                    # Compute metric
                    eval_score.append(rand_score(y_pred, seg))

            print(f'Epoch: {epoch + 1}\n'
                  f'\tTrain loss: {np.mean(train_loss)}\n'
                  f'\tTrain rand score: {np.mean(train_score)}\n\n'
                  f'\tEval loss: {np.mean(eval_loss)}\n'
                  f'\tEval rand score: {np.mean(eval_score)}')

            logger.add_scalar('train_loss', np.mean(train_loss), epoch)
            logger.add_scalar('train_score', np.mean(train_score), epoch)
            logger.add_scalar('eval_loss', np.mean(eval_loss), epoch)
            logger.add_scalar('eval_score', np.mean(eval_score), epoch)

        else:
            print(f'Epoch: {epoch + 1}\n'
                  f'\tTrain loss: {np.mean(train_loss)}\n'
                  f'\tTrain rand score: {np.mean(train_score)}')

            logger.add_scalar('train_loss', np.mean(train_loss), epoch)
            logger.add_scalar('train_score', np.mean(train_score), epoch)

        # Save model
        save_checkpoint(model)

        # lr_scheduler step
        scheduler.step()

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


def rand_score(y_pred, y_true):
    """Compute rand score with batch"""
    bs = y_pred.shape[0]
    score = 0

    for batch in range(bs):
        # Compute score for each batch
        pred = torch.round(y_pred[batch]).detach().cpu().numpy().astype(int).ravel()
        true = y_true[batch].detach().cpu().numpy().astype(int).ravel()

        score += m.adjusted_rand_score(pred, true)

    return score / bs


def save_checkpoint(model):
    torch.save(model.state_dict(), 'checkpoints/model.pt')
    return


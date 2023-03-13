from torch.utils.data import DataLoader
from data import CTDataset
from utils import train
from model import UNet


if __name__ == '__main__':
    lr = 0.001
    batch_size = 16
    num_epochs = 15

    train_loader = DataLoader(CTDataset('train'),
                              batch_size=batch_size,
                              shuffle=True)
    test_loader = DataLoader(CTDataset('test'),
                             batch_size=batch_size,
                             shuffle=False)

    model = UNet()

    train(model, train_loader, num_epochs=num_epochs, lr=lr)

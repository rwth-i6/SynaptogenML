from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets.mnist import MNIST

def create_mnist_dataloaders(batch_size):

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = MNIST('./', train=True, download=True,
                       transform=transform)
    dataset2 = MNIST('./', train=False,
                       transform=transform)
    dataloader_train = DataLoader(
        dataset=dataset1,
        batch_size=batch_size,
        shuffle=True,
    )

    dataloader_test = DataLoader(
        dataset=dataset2,
        batch_size=batch_size,
        shuffle=False,
    )
    return dataloader_train, dataloader_test



import numpy as np
from data import Data

import torch
from torch.utils.data import DataLoader, TensorDataset

if __name__ == "__main__":
    data = Data()

    train_images = torch.from_numpy(data.extract_prep_images("train"))
    train_labels = torch.from_numpy(data._extract_label("train"))

    test_images = torch.from_numpy(data.extract_prep_images("test"))
    test_labels = torch.from_numpy(data._extract_label("test"))

    # Config
    batch_size = 32
    num_epochs = 10
    step_per_epochs = train_images.shape[0] // batch_size

    # Pytorch train and test sets
    train = TensorDataset(train_images, train_labels)
    test = TensorDataset(test_images, test_labels)

    # Data loader
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    class MyModel(torch.nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()

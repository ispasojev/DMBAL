from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torchvision.transforms as transforms


class Cifar10Dataset(Dataset):

    def __init__(self, path):
        self.data = pd.read_csv(path)
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def len(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data.iloc[index, 1:].values.astype(np.uint8).reshape((1, 28, 28))
        label = self.data.iloc[index, 0]

        if self.transform is not None:
            image = self.transform(image)

        return image, label
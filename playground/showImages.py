import matplotlib.pyplot as plt
from src.data.DataLoader import DataLoader
from src.data.DataSetEnum import DataSetEnum
import numpy as np

# functions to show an image
import torchvision

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# get trainloader
dataloader = DataLoader()
dataloader.load_dataset(DataSetEnum.CIFAR10)
trainloader = dataloader.getTrainloader()

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
print("1")

# get some random training images
dataiter = iter(trainloader)
print("1.2")
images, labels = dataiter.next()
print("2")

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
print("3")
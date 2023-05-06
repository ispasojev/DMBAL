#%%
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from src.database.Database import Database

from src.data.DataLoader import DataLoader
from src.data.DataSetEnum import DataSetEnum
import torch.optim as optim
from src.model.Net import Net
import torch.nn as nn

#Set up db
if __name__ == '__main__':
    database = Database()
    cur = database.getCur()

#define imshow
#todo: imshow später auslagern
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# define classes
#todo: classes später auslagern (evtl. in DataLoader)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# get trainloader
dataloader = DataLoader()
dataloader.load_dataset(DataSetEnum.CIFAR10)
trainloader = dataloader.getTrainloader()
testloader = dataloader.getTestloader()
#%%

# initialize net
net = Net()
#%%

# define a loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#%%

# train a network
for epoch in range(2): # loop over the dataset multiple times
    print("epoch")
    print(epoch)
    running_loss = 0.0

    print(__name__)
    if __name__ == '__main__':
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

print('Finished Training')
#%%
if __name__ == '__main__':

    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

    #%%
    print("dataiter")
    dataiter = iter(testloader)
    print("dataiter2")
    images, labels = dataiter.next()

    # print images
    print("dataiter3")
    imshow(torchvision.utils.make_grid(images))
    print("dataiter4")
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    #%%
    #Reloading the model: (was not necessary)
    net.load_state_dict(torch.load(PATH))

    #%%
    outputs = net(images)

    #%%
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(4)))

    #%%

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

    #%%

    acc = round((correct / total), 3)
    cur.execute("UPDATE datasets SET accuracy = ? WHERE dataset_name = 'cifar10'", (acc,))
    cur.execute('SELECT * FROM datasets')
    data = cur.fetchall()
    print(data)

    #%%
    print("1")
    class_correct = list(0. for i in range(10))
    print("2")
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

    for i in range(10):
        acc = round((class_correct[i] / class_total[i]), 3)
        cur.execute("UPDATE cifar10_classes SET accuracy = ? WHERE class_name = ?", (acc, classes[i],))

    cur.execute('SELECT * FROM cifar10_classes')
    data = cur.fetchall()
    print(data)


    print("Finished training + test + accuracy ")

    #
    database.closeConnection()

    #%%
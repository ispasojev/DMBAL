# %%
import enum
import pathlib

import modAL as modAL
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from src.data.DataLoader import DataLoader
from src.data.DataSetEnum import DataSetEnum

import torch.optim as optim

from src.database.mlflow import MLFlowClient
from src.model.Resnet34CnnModel import Resnet34CnnModel

import torch.nn as nn

class TrainValTestEnum(enum.Enum):
   TRAIN = "training"
   VALIDATE = "valdidation"
   TEST = "test"


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def train(log=True):
    if log:
        database = MLFlowClient()

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    dataset = DataSetEnum.CIFAR10
    sample_size = 1000
    budget_size = 10000

    hyperparameters = {
        "dataset": dataset.value,
        "sample_size": sample_size
    }

    # get trainloader
    dataloader = DataLoader()
    dataloader.load_dataset(dataset, budget_size, sample_size)  # CIFAR10: 10 classes

    valloader = dataloader.getValloader()

    budget = dataloader.getBudget()
    budgetloader = dataloader.getBudgetloader(budget)

    sampleset = dataloader.getSampleset()
    sampleloader = dataloader.getSampleloader(sampleset)

    testloader = dataloader.getTestloader()

    # shape of just 1 example: (32, 32, 3)

    #todo: begins
    #   trying out different ways to cluster dataset with kmeans
    #sampleset_array = next(iter(sampleloader))[0].numpy()
    # len(sampleset_array) = 4 but sample_size = 10 ?

    #n1, n2, q1, q2 = sampleset_array.shape
    #new_sampleset_array = sampleset_array.reshape(sampleset_array.shape[0], -1)

    #y_pred = KMeans(n_clusters=3).fit(sampleset_array)
    #kmeans(sampleset, 3)
    #print(y_pred)

    #kmeans = MiniBatchKMeans(n_clusters=3)
    #kmeans.fit_predict(new_sampleset_array)
    # todo: end

    trainset = dataloader.getTrainset()

    # initialize net
    net = Resnet34CnnModel()

    # https://modal-python.readthedocs.io/en/latest/content/apireference/uncertainty.html
    # modAL.uncertainty.classifier_margin(net, sampleset)

    # define a loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    #Budget-Loop starts
    #Aus budget dann sample definieren

    while(len(budget)>0):
        print("hi")

        # train a network
        for epoch in range(2):  # loop over the dataset multiple times
            print("epoch")
            print(epoch)
            running_loss = 0.0

            print(__name__)
            for i, data in enumerate(sampleloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                if i % 20 == 0:
                    print(i)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

                # Reshape input for clustering
                # inputs.shape = (1000,3,32,32)
                # Currently, you have 4 dimension to your input data (batch size, channels, height, width) you need to flatten out your
                # images to two dimensions (number of images, channels* height* width)
                # src: https://stackoverflow.com/questions/57187680/how-to-resolve-valueerror-found-array-with-dim-4-estimator-expected-2
                inputs = inputs.reshape(1000, 3 * 32 * 32)

                #todo: calculate uncertainty on trainset (=budget)
                #modAL.uncertainty.classifier_margin(net, inputs)
                # Error: torch.nn.modules.module.ModuleAttributeError:
                # 'Resnet34CnnModel' object has no attribute 'predict_proba'

                #todo: prefilter to beta*k most informative examples

                #todo: clustering of beta*k most informative examples
                y_pred = KMeans(n_clusters=3).fit_predict(inputs)

                #todo: select k examples closest to the cluster centers
                #   -> will be added to sampleset, removed from trainset

                # validate classifier
                calculateAccuracy(valloader, net, TrainValTestEnum.VALIDATE)

                # updated sets and loaders
                budgetloader, sampleloader, sampleset = extendSample(dataloader, budget, sampleset, sample_size)

                # print("len(sampleset)")
                # print(len(sampleset))
                # print("len(trainset)")
                # print(len(trainset))

    print('Finished Training')
    # Budget-Loop ends

    if log:
        # create experiment
        run_id, output_path = database.init_experiment(hyper_parameters=hyperparameters)
        output_path = pathlib.Path(output_path)
        torch.save(net.state_dict(), output_path / './cifar_net.pth')

    # %%
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    acc = calculateAccuracy(testloader, net, TrainValTestEnum.TEST)

    class_correct = list(0. for i in range(10))
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

    print(data)
    print("Finished training + test + accuracy ")
    result = {"acc": acc}

    if log:
        # log all results
        database.finalise_experiment(result=result)

def calculateAccuracy(loader, net, step: TrainValTestEnum):
    print("entered test method")
    print("step.value")
    print(step.value)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Validation: Accuracy of the network on ' + step.value + ' images: %d %%' % (
            100 * correct / total))
    acc = round((correct / total), 3)
    return acc

def extendSample(dataloader: DataLoader, budget, old_sampleset, sample_size):
    #sample_size = len(old_sampleset) + sample_size
    budget_length = int(len(budget) - sample_size)
    if(len(budget) <=0):
        budget_length = 0
        sample_size = int(len(budget))
    #todo: randomsplit is wrong for extendeSample due to examples which are chosen due to their calculated uncertainty.
    #       !!!! REFACTOR !!!!
    budget, new_sampleset = torch.utils.data.random_split(budget, [budget_length, sample_size])
    sampleset = torch.utils.data.ConcatDataset([old_sampleset, new_sampleset])
    budgetloader = dataloader.getBudgetloader(budget)
    sampleloader = dataloader.getSampleloader(sampleset)
    return budgetloader, sampleloader, sampleset



if __name__ == '__main__':
    train(log=False)

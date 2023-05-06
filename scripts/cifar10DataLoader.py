
from src.data.DataLoader import DataLoader
from src.data.DataSetEnum import DataSetEnum


dataloader = DataLoader()
dataloader.load_dataset(DataSetEnum.CIFAR10)
trainset = dataloader.getTrainset()
trainloader = dataloader.getTrainloader()
testset = dataloader.getTestset()
testloader = dataloader.getTestloader()
print(trainset)
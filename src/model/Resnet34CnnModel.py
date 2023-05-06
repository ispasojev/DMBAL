""" DEPRECATED """
import torch.nn as nn
import torchvision.models as models
from src.data.ModelEnum import ModelEnum


# src: https://medium.com/swlh/classification-of-weather-images-using-resnet-34-in-pytorch-7e86b2b24dcf

class Resnet34CnnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = models.resnet34(pretrained=False)
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 10)
        self.feature_extractor = nn.Sequential(*list(self.network.children())[:-1])
        self.model_type = ModelEnum.RESNET34

    def forward(self, xb):
        # predictions = self.network(xb)
        # feature_extractor  = nn.Sequential(*list(self.network.children())[:-1])
        representations = self.feature_extractor(xb)
        predictions = self.network(xb)
        return predictions, representations

    def freeze(self):
        # To freeze the residual layers
        for param in self.network.parameters():
            param.require_grad = False
        for param in self.network.fc.parameters():
            param.require_grad = True

    def unfreeze(self):
        # Unfreeze all layers
        for param in self.network.parameters():
            param.require_grad = True

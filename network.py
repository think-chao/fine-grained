import torch.nn as nn
import torch
import torchvision.models as models
import os

class DogModel (nn.Module):
    def __init__(self, n_class = 120):
        super(DogModel, self).__init__()
        self.cov = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=4),
            nn.BatchNorm2d(96),
            nn.ReLU(True),

            nn.MaxPool2d(3, stride =2),

            nn.Conv2d(96, 256, 5, padding=3),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.MaxPool2d(3, stride =2),

            nn.Conv2d(256, 384, 3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(True),

            nn.Conv2d(384, 384, 3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(True),

            nn.Conv2d(384, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.MaxPool2d(3, stride =2),
        )
        self.cls = nn.Sequential(
            nn.Linear(9216, 4096),
            nn.Linear(4096, 4096),
            nn.Linear(4096, n_class)
        )
        self.n_class = n_class


    def forward(self, input):
        batch = input.size(0)
        feature = self.cov(input)
        x = feature.view(batch, 9216)
        output = self.cls(input)
        return output


if __name__ == '__main__':
    input = torch.ones((1, 3, 299, 299))
    vgg = DogModel()
    vgg(input)



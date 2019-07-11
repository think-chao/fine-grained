import torch.nn.functional
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import time
import random
from Dogdataset import DDS
from torch.utils.data import DataLoader
import torchvision
from models.DFL import DFL_VGG16
from visdom import Visdom
from utils.cal_time import *

train_data = '/home/ai/Documents/datasets/dog/train'
testing_data = '/home/ai/Documents/datasets/dog/val'
label_data = '/home/ai/Documents/datasets/dog/labels.csv'
checkpoints = './checkpoints/'
batch_size = 6
epochs = 20
lr = 0.001
use_gpu = True
df = pd.read_csv(label_data)
breed_arr = sorted(list(set(df['breed'].values)))
num_classes = 120


def train(trainLoader, testLoader):
    model = DFL_VGG16()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # use smaller lr for pre-trained models
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=4, verbose=True)

    # if len(os.listdir('./checkpoints')) > 0:
    #     print('resume model')
    #     checkpoint = torch.load(os.path.join(checkpoints, checkpoints + str(
    #         max([file.split('.')[0].split('_')[-1] for file in os.listdir(checkpoints)])) + '.pth'))
    #     model.load_state_dict(checkpoint['model'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     scheduler.load_state_dict(checkpoint['scheduler'])

    if torch.cuda.is_available() and use_gpu:
        print('GPU ready')
        model = model.cuda()

    for epoch in range(epochs):
        model.train()
        epoch_loss = []
        for iter, sample in enumerate(trainLoader):
            if iter == len(trainLoader) - 1:
                continue
            input = sample['Image'].cuda() if use_gpu else sample['Image']  # size of batch_size
            label = sample['IntLabel'].cuda() if use_gpu else sample['IntLabel']
            optimizer.zero_grad()  # clears

            out1, out2, out3, _ = model(input)
            out = out1 + out2 + 0.1 * out3

            loss = criterion(out, label)

            score = torch.max(out, 1)[1]  # array of the indexes of the max output number in each row
            correct = int(torch.sum(score == label))

            print('epoch {} = iteration {} lr {} == > loss {:.3f} total {} accuracy {:.3f}% correct {}'.format(
                iter, iter, optimizer.state_dict()['param_groups'][0]['lr'], loss.data, batch_size,
                100 * correct / len(input), correct))
            loss.backward()

        val_loss = eval(model, testLoader, criterion)
        epoch_loss.append(val_loss)
        scheduler.step(val_loss)

        if epoch == 0 or val_loss < min(epoch_loss[:-1]):
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, './checkpoints/' + 'dfl_' + str(val_loss) + '.tar')
    print('Done solving!')


def eval(m, tl, crt):
    m.eval()
    valloss = 0
    total, correct = 0, 0

    for i, sample in enumerate(tl):
        if i == len(tl) - 1:
            continue
        input = sample['Image'].cuda() if use_gpu else sample['Image']  # size of batch_size
        total += input.size(0)
        label = sample['IntLabel'].cuda() if use_gpu else sample['IntLabel']

        out1, out2, out3, _ = m(input)
        out = out1 + out2 + 0.1 * out3
        correct += int(torch.sum(torch.max(out, 1)[1] == label))
        valloss += crt(out, label).data

    print('Eval: total:{} correct:{}'.format(total, correct))
    return valloss


if __name__ == '__main__':
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.RandomRotation(30),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomCrop((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        # torchvision.transforms.TenCrop((299,299)),
        torchvision.transforms.ToTensor()
    ])
    train_set = DDS(train_data, label_data, train_transform)
    testing_set = DDS(testing_data, label_data, test_transform)
    train_dataLoader = DataLoader(train_set, batch_size, shuffle=True)
    testing_dataLoader = DataLoader(testing_set, batch_size)
    train(train_dataLoader, testing_dataLoader)

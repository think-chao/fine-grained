from Dogdataset import DDS
from torch.utils.data import DataLoader
from network import DogModel
import torchvision
import torch.nn as nn
import torch.nn.functional
import torch
import pandas as pd
import numpy as np
import os
import time
import random
import matplotlib.pyplot as plt
from PIL import Image
from thop import profile
import torchvision.models as models
from visdom import Visdom

from utils.cal_time import *

train_data = '/home/ai/Documents/datasets/dog/train'
testing_data = '/home/ai/Documents/datasets/dog/val'
label_data = '/home/ai/Documents/datasets/dog/labels.csv'
checkpoints = './checkpoints/'
batch_size = 6
epoch = 20
lr = 0.001
use_gpu = True
df = pd.read_csv(label_data)
breed_arr = sorted(list(set(df['breed'].values)))
num_classes = 120


def dfl_train(tl):
    pass


def train(tl, vl):
    # viz = Visdom(env='test')
    x, y = 0, 0
    # win = viz.line(X=np.array([x]), Y=np.array([y]))
    criterion = nn.CrossEntropyLoss()
    model = models.resnet152(pretrained=True)  # ---> input size must be 3x299x299
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # use smaller lr for pre-trained models
    model.fc = nn.Linear(2048, num_classes)

    if len(os.listdir('./checkpoints')) > 0:
        print('resume model')
        checkpoint = torch.load(os.path.join(checkpoints, checkpoints + str(
            max([file.split('.')[0].split('_')[-1] for file in os.listdir(checkpoints)])) + '.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    if torch.cuda.is_available() and use_gpu:
        print('GPU ready')
        model = model.cuda()
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1)
    accuracy_test = []
    for i in range(epoch):
        model.train()
        for iter, sample in enumerate(tl):  # iterates len(database)/batch_size
            # tic()
            if iter == len(tl) - 1:
                continue
            input = sample['Image'].cuda() if use_gpu else sample['Image']  # size of batch_size
            label = sample['IntLabel'].cuda() if use_gpu else sample['IntLabel']
            optimizer.zero_grad()  # clears
            output = model(input)  # output is a batch_size x n_class 2d arr
            score = torch.max(output, 1)[1]  # array of the indexes of the max output number in each row
            correct = int(torch.sum(score == label))
            loss = criterion(output, label)
            # viz.line(X=np.array([iter]), Y=np.array([float(loss)]), win=win, update='append')
            print('epoch {} = iteration {} lr {} == > loss {:.3f} total {} accuracy {:.3f}% correct {}'.format(
                i, iter, optimizer.state_dict()['param_groups'][0]['lr'], loss.data, batch_size,
                100 * correct / len(input), correct))
            loss.backward()
            optimizer.step()
        # toc()
        val_info = val(vl, model)
        eval_acc = val_info["Accuracy"]
        total_correct = val_info["Correct"]
        print('=====> eval accuracy :{}'.format(eval_acc))
        accuracy_test.append(eval_acc)
        if i == 0 or accuracy_test[-1] > max(accuracy_test[:-1]):
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, './checkpoints/' + 'res_'+str(eval_acc)+ '.tar')


def val(vl, m):
    m.eval()  # use when trying to compare label and predictor
    total = 0
    count = 0
    for iter, sample in enumerate(vl):
        if iter == len(vl) - 1:
            continue
        input = sample['Image'].cuda()
        total += input.size(0)
        # img_name = sample['Name']
        # str_label = sample['Breed']
        label = sample['IntLabel'].cuda()
        output = m(input)
        score = torch.max(output, 1)[1]
        correct = int(torch.sum(score == label))
        count += correct
    info = {"Accuracy": count / total * 100, "Correct": count, "Total": total}
    return info


def main():
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


main()

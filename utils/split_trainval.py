from tqdm import tqdm
import os
import random
import shutil

root_path = '/home/ai/Desktop/project2/datasets/train'
label_path = '/home/ai/Desktop/project2/datasets/labels.csv'


train_save = '/home/ai/Documents/datasets/dog/train'
val_save = '/home/ai/Documents/datasets/dog/val'

all_img = os.listdir(root_path)
random.shuffle(all_img)

trainval_ratio = 0.7
train_img = all_img[:int(trainval_ratio*len(all_img))]
val_img = all_img[int(trainval_ratio*len(all_img)):]

for img in tqdm(train_img):
    shutil.copy(os.path.join(root_path, img), os.path.join(train_save, img))

for img in tqdm(val_img):
    shutil.copy(os.path.join(root_path, img), os.path.join(val_save, img))




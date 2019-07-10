from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import pandas as pd


class DDS(Dataset):
    def __init__(self, path, label_path, transform = None):
        super(DDS, self).__init__()
        self.path = path
        self.label_path = label_path
        self.transform = transform
        self.csv_file = pd.read_csv(label_path)

    def __len__(self):
        return len(os.listdir(self.path))

    def __getitem__(self, item):
        files = os.listdir(self.path)
        file = files[item]
        key = file.split(".")[0]
        str = os.path.join(self.path, file)
        label = self.csv_file[self.csv_file['id'] == key]['breed'].values[0]
        breed = sorted(list(set(self.csv_file['breed'].values)))
        intlabel = breed.index(label) # maps string label to an int
        im = Image.open(str)
        if self.transform:
            im = self.transform(im)
        dict = {'IntLabel': intlabel, 'Image': im, 'Name': str, 'Breed': label}
        return dict


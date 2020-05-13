from torch.utils.data import Dataset
from torchvision import transforms
import os
import torch
from PIL import Image
IMG_SIZE = (64,64)

class TinyImNet(Dataset):

    def __init__(self, path, is_train, train_path):
        self.path = path
        self.is_train = is_train
        self.train_path = train_path
        self.to_tensor = transforms.ToTensor()
        self.keys = self.get_keys()
        if is_train:
            self.labels, self.list_ID = self.get_train()
        else:
            self.labels = self.val_labels()
            self.list_ID = self.val_path()


    def __len__(self):
        return len(self.list_ID)

    def __getitem__(self, index):
        ID = self.list_ID[index]
        img = Image.open(ID)
        X = self.to_tensor(img)
        y = self.keys[self.labels[index]]
        if list(X.size()) == [3, 64, 64]:
            return X,y
        else:
            print('Error Loading photo')
            X = torch.ones([3, 64, 64])
            y = 00
            return X, y

    def get_keys(self):
        dic, count = {}, 0
        for class_name in os.listdir(self.train_path):
            dic[class_name] = count
            count += 1
        return dic

    def get_train(self):
        labels = []
        paths = []
        for class_name in os.listdir(self.path):
            class_path = os.path.join(self.path, class_name + '/images')
            for image_name in os.listdir(class_path):
                img_path = os.path.join(class_path, image_name)
                labels.append(class_name)
                paths.append(img_path)

        return labels, paths

    def val_labels(self):
        labels = []
        with open(os.path.join(self.path, 'val_annotations.txt')) as f:
            lines = f.readlines()
        for l in lines:
            label = l.split('	')[1]
            labels.append(label)
        return labels

    def val_path(self):
        paths = []
        for img_name in os.listdir(self.path+ '/images/'):
            path = self.path + '/images/' + img_name
            paths.append(path)
        return paths
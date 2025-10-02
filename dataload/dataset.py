import os
import random
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod
import torchvision
import cv2
from sklearn.decomposition import NMF
import pandas as pd
import cv2
import json
class BaseDataset(data.Dataset):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, path, ann_txt, transform=None):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.ann_txt = ann_txt
        self.path = path
        self.img_label = self.load_txt()
        self.img = [os.path.join(self.path, img) for img in list(self.img_label.keys())]
        self.label = [label for label in list(self.img_label.values())]
        self.transform = transform

    def load_txt(self):
        data_infos = {}
        with open(self.ann_txt, 'r') as f:
            samples = [line.strip().split(' ') for line in f.readlines()]
            for filename, labels in samples:
                data_infos[filename] = np.array(labels, dtype=np.int64)
        return data_infos


    @abstractmethod
    def __getitem__(self, index):
        image = Image.open(self.img[index])
        label = self.label[index]
        label = torch.from_numpy(np.array(label))
        if self.transform is not None:
            image = self.transform(image)

       # trans_totensor = torchvision.transforms.ToTensor()
      #  image_tensor = trans_totensor(image)
        return {'label': label, 'image': image}

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.img)

class EyePACS(data.Dataset):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, path, ann_txt, transform=None):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.ann_txt = ann_txt
        self.path = path
        self.labels = pd.read_csv(ann_txt)
       # self.img = [os.path.join(self.path, img) for img in list(self.img_label.keys())]
        #self.label = [label for label in list(self.img_label.values())]
        self.transform = transform

    def load_csv(self):
        data_infos = {}


    @abstractmethod
    def __getitem__(self, idx):
        img_name = self.labels.iloc[idx, 0]  # First column contains image names
        label = int(self.labels.iloc[idx, 1])  # Second column contains labels

        # Load the image from disk
        img_path = os.path.join(self.path, img_name + '.png')  # Adjust extension if necessary
        image = Image.open(img_path).convert('RGB')  # Convert to RGB for consistency

        # Apply transformations if provided
        if self.transform is not None:
            image = self.transform(image)
        else:
            trans_totensor = torchvision.transforms.ToTensor()
            image = trans_totensor(image)
        return {'label': label, 'image': image}

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.labels)


class IDRiD(data.Dataset):
    """Dataset class that loads images and labels from a CSV file.

        CSV file format:
        Image name, Retinopathy grade
        IDRiD_001, 4
        IDRiD_002, 3
        ...
        """

    def __init__(self, img_dir, csv_file, transform=None, img_suffix=".jpg"):
        """
        Args:
            img_dir (str): Path to image directory
            csv_file (str): Path to CSV file with 'Image name' and 'Retinopathy grade'
            transform (callable, optional): Transform to apply to images
            img_suffix (str): Image file extension (default: '.png')
        """
        self.img_dir = img_dir
        self.csv_file = csv_file
        self.transform = transform
        self.img_suffix = img_suffix

        # load csv
        df = pd.read_csv(self.csv_file)
        self.img_names = df.iloc[:, 0].values  # 第一列：文件名
        self.labels = df.iloc[:, 1].values  # 第二列：标签

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.img_names[idx] + self.img_suffix)
        image = Image.open(img_name).convert("RGB")
        label = int(self.labels[idx])  # 转为整数

        if self.transform:
            image = self.transform(image)

        return {"image": image, "label": torch.tensor(label, dtype=torch.long)}


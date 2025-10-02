import torch
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod
import torchvision
import pandas as pd
import numpy as np
import json
import os
import warnings
class MyDataset(data.Dataset):
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
        suffix = '.png'
        img_name = os.path.join(self.path,
                                str(self.labels.iloc[idx, 0]) + suffix)
        label = self.labels.iloc[idx, 1:].values
        label = label.astype('double')

        try:
            image = Image.open(img_name).convert('RGB')

        except OSError as e:
            print(f"Truncated image file: {img_name}, Error: {e}")
            raise Exception(f"Truncated image file: {img_name}") from e

        except Exception as e:

            print(f"Error opening image: {img_name}, Error: {e}")
            raise Exception(f"Error opening image: {img_name}") from e

        cls = self.labels.columns.values[1:]
        class_indices = dict((k, v) for v, k in enumerate(cls))
        json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
        with open('class_indices.json', 'w') as json_file:
            json_file.write(json_str)

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








import os
from typing import Tuple
from torch.utils.data import *
import numpy as np
from PIL import Image
import torch

class WaterBodyDataset(Dataset):

    def __init__(self, root:str, transform = None):
        self._image_data = os.path.join(root, "Images")
        self._mask_data = os.path.join(root, "Masks")
        self._filenames = os.listdir(self._image_data)
        self.transform = transform
        self._length = len(self._filenames)
        
    def __len__(self):
        return self._length
    
    def __getitem__(self, index) -> Tuple:
        
        img_path = os.path.join(self._image_data, self._filenames[index])
        mask_path = os.path.join(self._mask_data, self._filenames[index])

        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        ## TODO Check other solutions
        #mask = mask[:,:,0]
        
        #mask[mask==255] = 1.0
        
        mask[mask<128] = 0.0
        mask[mask>=128] = 1.0
        
        #mask[mask<200] = 0.0
        #mask[mask>0.0] = 1.0


        #using albumentations
        if self.transform:
            augmentations = self.transform(image=img, mask=mask)
            img = augmentations["image"]
            mask = augmentations["mask"]

        #mask = mask.float()
        #print(mask)
        #print(img.max(), mask.max(),mask.shape)    
        return (img, mask)


class CarvanaDataset(Dataset):
    def __init__(self, root:str, transform=None):
        self.image_dir = os.path.join(root, "Images")
        self.mask_dir = os.path.join(root, "Masks")
        self.transform = transform
        self.images = os.listdir(os.path.join(root, "Images"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        
        return image, mask

class WaterBodyGeneratorDataset(torch.utils.data.Dataset):
    """Some Information about WaterBodyGeneratorDataset"""
    def __init__(self, root, transform=None):
        super(WaterBodyGeneratorDataset, self).__init__()

        self._image_data = os.path.join(root, "Images")
        self._filenames = os.listdir(self._image_data)
        self.transform = transform
        self._length = len(self._filenames)
    def __getitem__(self, index):
        img_path = os.path.join(self._image_data, self._filenames[index])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        
        return img

    def __len__(self):
        return self._length
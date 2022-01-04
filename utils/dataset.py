import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import cv2

import numpy as np


class URISC(Dataset):
    def __init__(
        self, 
        dir: str, 
        mode: str = 'train',
        transform: Optional[Callable] = None, 
        data_rank: str = 'simple',
    ):
        super(URISC, self).__init__()
        self.dir = dir
        self.mode = mode
        self.transform = transform
        self.data_rank = data_rank

        if data_rank == 'simple':
            self.transform_normalize = transforms.Normalize(mean=0.520, std=0.185)
        elif data_rank == 'complex':
            self.transform_normalize = transforms.Normalize(mean=0.518, std=0.190)
        self.transform_totensor = transforms.ToTensor()

        self.ids = [os.path.join(dir, data_rank, mode, filename) for filename in os.listdir(os.path.join(dir, data_rank, mode))]
        if not self.ids:
            raise RuntimeError(f'No input file found in {os.path.join(dir, data_rank, mode)}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image = cv2.imread(self.ids[idx])
        # print(image.shape)
        
        if self.mode == 'test':
            if self.transform is not None:
                image = self.transform(image=image)
            return image.float().contiguous(), self.ids[idx]
        
        mask_path = self.ids[idx].replace(self.mode, "label/"+self.mode)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # print(mask)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            transformed_image = transformed['image']
            transformed_mask = transformed['mask']
        else:
            transformed_image = image
            transformed_mask = mask

        transformed_image = self.transform_totensor(transformed_image)
        transformed_image = self.transform_normalize(transformed_image)
        transformed_mask = self.transform_totensor(transformed_mask)

        # transformed_image = np.transpose(transformed_image, (2, 0, 1))
        # transformed_mask = np.expand_dims(transformed_mask, axis=0)

        return transformed_image, transformed_mask
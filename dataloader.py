
import cv2
import torch

import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import Dataset, DataLoader

class VAE_Loader(Dataset):
    def __init__(self, data, train, k):
        self.data = data
        self.train = train
        self.k = k

        #self.means = [(0.3559621),(0.3573706),(0.3527124), (0.3561953), (0.3515642)]
        #self.stds = [(0.17658396),(0.18102418),(0.17620215),(0.1776245),(0.18114814)]

        if self.train:
            self.transforms = A.Compose([
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.HorizontalFlip(p=0.5),A.RandomBrightnessContrast(p=0.5),    
            A.ColorJitter(), 
         #   A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
            ToTensorV2()]) 
        else:
           self.transforms = A.Compose([
                                    #    A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
                                        ToTensorV2()
                             ])
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        path = self.data['Path'].values[idx]

        arr = cv2.imread(path,0)
        arr = cv2.resize(arr, (256,256), interpolation = cv2.INTER_AREA)
        arr = np.expand_dims(arr, -1)
        arr = arr/np.max(arr)
        
        arr = arr.astype(np.float32)

        image = self.transforms(image=arr)["image"]

        y_label = self.data[['No Finding', 'Lung Opacity', 'Pleural Effusion', 'Support Devices']].values[idx] #[['No Finding', 'Enlarged Cardiomediastinum','Cardiomegaly','Lung Lesion', 'Lung Opacity','Edema','Consolidation','Pneumonia', 'Atelectasis','Pneumothorax','Pleural Effusion','Pleural Other','Fracture','Support Devices']]
        y_label = [0 if i==-1 or i==0 else 1 for i in y_label]
        y_label = torch.tensor(y_label, dtype=torch.float)
        y_label = torch.unsqueeze(y_label, 0)

        return image, y_label
    
def make_dataloaders(train_df, val_df, test_df, params):
    train_dataset = VAE_Loader(train_df, train=True, k=params["k"])
    train_dg = DataLoader(train_dataset, batch_size=params["batchsize"], 
    shuffle=True, num_workers = params["num_workers"], pin_memory=True, drop_last=True)
        
    val_dataset = VAE_Loader(val_df, train=False, k=params["k"])
    val_dg = DataLoader(val_dataset, batch_size=params["batchsize"], 
    shuffle=True, num_workers = params["num_workers"], pin_memory=True, drop_last=True)

    test_dataset = VAE_Loader(test_df, train=False, k=params["k"])
    test_dg = DataLoader(test_dataset, batch_size=params["batchsize"], 
    shuffle=False, num_workers = params["num_workers"], pin_memory=True, drop_last=True)

    return {'train':train_dg, 'val':val_dg, 'test':test_dg}
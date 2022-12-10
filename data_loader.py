import torch.nn.functional as F
import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image


class scattering_dataloader(torch.utils.data.Dataset):


    def __init__(self, directory,typei='gt' ):
        super(scattering_dataloader, self).__init__()

        self.directory= directory
        self.name_list = os.listdir(self.directory)

        self.typei = typei

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()


        file_name = self.name_list[idx]
        sample = np.load(os.path.join(self.directory,file_name))

        if self.typei == 'gt':

            uu = np.expand_dims(sample , axis = 0)
            uu = (uu-1540.0)/(5750.0-1540.0)
            sampli = torch.tensor(uu, dtype = torch.float32)

        elif self.typei == 'measure':

            sampli = np.zeros((2,np.shape(sample)[0],np.shape(sample)[1]))
            sampli[0]= sample.real
            sampli[1] = sample.imag
            sampli = torch.tensor(sampli/400000.0, dtype = torch.float32)
            sampli = F.interpolate(sampli[None, ...], 128 , mode = 'bilinear', align_corners=False)[0]
            # sampli = sampli.repeat(1,8,8)

        elif self.typei[0:2] == 'bp':
            uu = np.expand_dims(sample , axis = 0)
            uu = (uu-(-2e-6))/(2e-6-(-2e-6))
            sampli = torch.tensor(uu, dtype = torch.float32)            

        return sampli
        


class general_dataloader(torch.utils.data.Dataset):
    def __init__(self, dataset = 'mnist', size=(32,32), c = 1):
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
        ])

        self.c = c
        self.dataset = dataset
        if self.dataset == 'mnist':
            self.img_dataset = torchvision.datasets.MNIST('data/MNIST', train=True,
                                                    download=True)
        
        elif self.dataset == 'celeba-hq':
            celeba_path = '/raid/Amir/Projects/datasets/celeba_hq/celeba_hq_256/'
            self.img_dataset = ImageFolder(celeba_path, self.transform)
            
            

    def __len__(self):
        return len(self.img_dataset)

    def __getitem__(self, item):
        img = self.img_dataset[item][0]
        if self.dataset == 'celeba-hq':
            img = transforms.ToPILImage()(img)

        img = self.transform(img)

        return img
    
  

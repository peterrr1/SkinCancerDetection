from PIL import Image
import pandas as pd
from torchvision import transforms as v2
import torch
from torch import nn
import numpy as np
from matplotlib import pyplot as plt
from skimage.restoration import denoise_nl_means, estimate_sigma
from visualize import show_images
from skimage import exposure, filters
from skimage import img_as_float
from skimage import segmentation




class ProcessDataset:
    def __init__(self, data_path, label_path):
        self.data_path = data_path
        self.label_path = label_path

        self.transform = v2.Compose([
            v2.Resize(size=(224, 224), antialias=True),
            v2.RandomRotation(30),
            v2.RandomHorizontalFlip(p=0.3),
            v2.Grayscale(num_output_channels=1),
            v2.PILToTensor(),
            ScaleAndConvert(),
            HistogramEqualization(),
            v2.Normalize((0.485,), (0.229,)),
        ])

        self.samples = self._create_dataset()


    def __len__(self):
        return len(self.samples)


    def _create_dataset(self):
        np.random.seed(0)
        csv_file = pd.read_csv(self.label_path, dtype={'isic_id': str, 'target': int})[['isic_id', 'target']]
        pos_samples = csv_file[csv_file['target'] == 1].values
        neg_samples = csv_file[csv_file['target'] == 0].values

        idxs = np.random.randint(0, len(neg_samples), size=len(pos_samples))

        neg_samples = neg_samples[idxs, :]

        samples = np.concatenate((pos_samples, neg_samples))
        samples = np.array(sorted(samples, key = lambda row: row[0]))

        return samples
    

    def __getitem__(self, idx):
        image, label = self.samples[idx]
        image = Image.open(f'{self.data_path}/{image}.jpg')
        image = self.transform(image)
        label = torch.tensor(label)

        return image, label




class Denoise(nn.Module):
    def __init__(self, h_weight: float = 100, patch_size: int = 5,
                 patch_distance: int = 6, fast_mode: bool = False):
        super(Denoise, self).__init__()

        self.h_weight = h_weight
        self.patch_kw = dict(
            fast_mode = fast_mode, patch_size = patch_size, patch_distance = patch_distance, channel_axis = -1
        )

    def forward(self, x):
        est_sigma = np.mean(estimate_sigma(x, channel_axis=-1))
        x = denoise_nl_means(x, h = self.h_weight * est_sigma, sigma = est_sigma, **self.patch_kw)

        return torch.tensor(x)




class ScaleAndConvert(nn.Module):
    def __init__(self):
        super(ScaleAndConvert, self).__init__()
    
    def forward(self, x):
        x = np.array(x.squeeze())
        x = img_as_float(x)
        
        return x
    



class HistogramEqualization(nn.Module):
    def __init__(self):
        super(HistogramEqualization, self).__init__()
    
    def forward(self, x):
        x = exposure.equalize_hist(x)
        return torch.tensor(x).unsqueeze(0)









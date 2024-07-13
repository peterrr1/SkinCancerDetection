from PIL import Image
import pandas as pd
from torchvision import transforms as v2
import torch
from torch import nn
import numpy as np
from matplotlib import pyplot as plt
from skimage.restoration import denoise_nl_means, estimate_sigma
from visualize import show_images


img = Image.open('/Users/peterbrezovcsik/Documents/Kaggle/SkinCancerDetection/data/train-image/image/ISIC_0082829.jpg')

class ProcessDataset:
    def __init__(self, data_path, label_path):
        self.data_path = data_path
        self.label_path = label_path

        self.transform = v2.Compose([
            v2.Grayscale(num_output_channels=3),
            v2.PILToTensor(),
            v2.ConvertImageDtype(torch.float32) ,
            Denoise(h_weight=0.8),
            v2.Normalize(mean=[0.485,], std=[0.229,]),
        ])

        self.samples = self._create_dataset()

    

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
    def __init__(self, h_weight: float = 0.8, patch_size: int = 5,
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



def main():
    
    ds = ProcessDataset(data_path='/Users/peterbrezovcsik/Documents/Kaggle/SkinCancerDetection/data/train-image/image',
                        label_path='/Users/peterbrezovcsik/Documents/Kaggle/SkinCancerDetection/data/train-metadata.csv')


main()










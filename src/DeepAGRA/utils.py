from typing import Optional

import torchvision.transforms as transforms
from torchvision.datasets.folder import pil_loader

from wrench.dataset import TorchDataset, ImageDataset


class ResNetDataset(TorchDataset):
    def __init__(self, dataset: ImageDataset, n_data: Optional[int] = 0):
        super(TorchDataset, self).__init__(dataset, n_data)
        self.preload_image = dataset.preload_image
        if self.preload_image:
            self.images = dataset.images
        self.input_size = dataset.image_input_size
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ])

    def __getitem__(self, idx):
        idx = idx % self.n_data_
        if self.preload_image:
            img = self.images[idx]
        else:
            img = pil_loader(self.data[idx]['image_path']).resize(self.input_size)
        img = self.transform(img)
        d = {
            'ids': idx,
            'labels': self.labels[idx],
            'weak_labels': self.weak_labels[idx],
            'image': img,
        }
        if self.features is not None:
            d['features'] = self.features[idx]
        return d

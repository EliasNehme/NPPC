import os
import random

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision


## Dataset Auxiliary
## =================
def find_data_folder(data_folders, folder_name):
    if isinstance(data_folders, str):
        data_folders = (data_folders,)
    data_folder = None
    for data_folder_tmp in data_folders:
        data_folder_tmp = os.path.join(data_folder_tmp, folder_name)
        if os.path.isdir(data_folder_tmp):
            data_folder = data_folder_tmp
            break
    if data_folder is None:
        raise Exception('Could not fond the data in any of the provided folders')
    return data_folder

def split_dataset(dataset, split_size, rand=True):
    n_samples = len(dataset)
    if rand:
        indices = np.random.RandomState(42).permutation(n_samples)  # pylint: disable=no-member
    else:
        indices = np.arange(n_samples)
    indices1 = indices[:-split_size]
    indices2 = indices[-split_size:]
    dataset1 = torch.utils.data.Subset(dataset, indices1)
    dataset2 = torch.utils.data.Subset(dataset, indices2)
    return dataset1, dataset2


def split_batch(batch, n):
    if isinstance(batch, (tuple, list)):
        batches = tuple(zip(*[split_batch(batch_, n) for batch_ in batch]))
    else:
        batches = torch.chunk(batch, n, dim=0)
    return batches


class ImageFilesDataset(torch.utils.data.Dataset):
    def __init__(self, filenames, transform=None):
        super().__init__()

        if isinstance(filenames, str):
            filenames = [os.path.join(filenames, filename) for filename in np.sort(os.listdir(filenames))]
        self.filenames = filenames
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        img = Image.open(filename).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img


class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, store_dataset=False, fixed_size_tensor=True, device=None, transform=None):
        super().__init__()

        self.dataset = dataset
        self.transform = transform

        if not store_dataset:
            self.stored = None
            self.is_stored = None
        elif fixed_size_tensor:
            x = self.dataset[0]
            self.stored = torch.zeros((len(self),) + x.shape, dtype=x.dtype, device=device)
            self.is_stored = torch.zeros((len(self),), dtype=torch.long)
        else:
            self.stored = [None] * len(self)
            self.is_stored = torch.zeros((len(self),), dtype=torch.long)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.stored is None:
            x = self.dataset[index]
        elif self.is_stored[index] == 0:
            x = self.dataset[index]
            self.stored[index] = x
            self.is_stored[index] = 1
        else:
            x = self.stored[index]

        if self.transform is not None:
            x = self.transform(x)
        return x


class GetIndex(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index
    
    def forward(self, imgs):
        return imgs[self.index]

    def __repr__(self):
        return self.__class__.__name__ + f'(index={self.index})'


class PairsDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets, transform=None):
        super().__init__()

        self.datasets = datasets
        self.transform = transform

    def __len__(self):
        return min([len(dataset) for dataset in self.datasets])

    def __getitem__(self, index):
        x = tuple([dataset[index] for dataset in self.datasets])
        return x


def crop_scaled_pair(hr_img, lr_img, patch_size, method='rand'):
    hr_width, _ = hr_img.size
    lr_width, lr_height = lr_img.size

    scale = hr_width // lr_width
    lr_patch_size = patch_size // scale
    if method.lower() == 'rand':
        left = random.randrange(0, lr_width - lr_patch_size + 1)
        top = random.randrange(0, lr_height - lr_patch_size + 1)
    elif method.lower() == 'center':
        left = (lr_width - lr_patch_size) // 2
        top = (lr_height - lr_patch_size) // 2
    else:
        raise Exception(f'Unsuported method type: "{method}"')
    right = left + lr_patch_size
    bottom = top + lr_patch_size

    lr_patch = lr_img.crop((left, top, right, bottom))

    left *= scale
    top *= scale
    right *= scale
    bottom *= scale
    hr_patch = hr_img.crop((left, top, right, bottom))

    return hr_patch, lr_patch


class CropScaledPair(nn.Module):
    def __init__(self, patch_size=None, method='rand'):
        super().__init__()
        self.patch_size = patch_size
        self.method = method
    
    def forward(self, imgs):
        imgs = crop_scaled_pair(imgs[0], imgs[1], patch_size=self.patch_size, method=self.method)
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + f'(patch={self.patch_size}, method={self.method.lower()})'


## Datasets
## ========
class MNISTDataModule(object):
    shape = (1, 28, 28)
    # mean = None
    # std = None
    mean = 0.5
    std = 0.2

    def __init__(self, data_folder, n_valid=256, rand_valid=True, remove_labels=False, store_dataset=False, device='cpu'):  # pylint: disable=abstract-method
        super().__init__()


        data_folder = find_data_folder(data_folder, 'MNIST')
        print(f'Loading data from: {data_folder}')

        ## Base dataset
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            ])
        train_set = torchvision.datasets.MNIST(root=os.path.dirname(data_folder), train=True, transform=transform)
        test_set = torchvision.datasets.MNIST(root=os.path.dirname(data_folder), train=False, transform=transform)

        ## Remove labels
        if remove_labels:
            train_set = DatasetWrapper(train_set, transform=GetIndex(0))
            test_set = DatasetWrapper(test_set, transform=GetIndex(0))
            ## Store dataset
            if store_dataset:
                train_set = DatasetWrapper(train_set, store_dataset=True, device=device)
                test_set = DatasetWrapper(test_set, store_dataset=True, device=device)

        ## Split
        if n_valid != 0:
            train_set, valid_set = split_dataset(train_set, n_valid, rand=rand_valid)
        else:
            valid_set = test_set

        ## set datasets
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set


class CelebAHQ256DataModule(object):
    mean = 0.5
    # std = 0.2
    std = 0.5

    def __init__(self, img_size, data_folder, store_dataset=False):
        super().__init__()

        self.img_size = img_size
        self.shape = (3, self.img_size, self.img_size)

        data_folder = find_data_folder(data_folder, 'CelebAMask-HQ-256')
        print(f'Loading data from: {data_folder}')

        ## Base dataset
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.img_size, interpolation=torchvision.transforms.InterpolationMode.BOX),
            torchvision.transforms.ToTensor(),
            ])

        ## Base dataset
        train_set = ImageFilesDataset(os.path.join(data_folder, 'train'), transform=transform)
        valid_set = ImageFilesDataset(os.path.join(data_folder, 'valid'), transform=transform)
        test_set = ImageFilesDataset(os.path.join(data_folder, 'test'), transform=transform)

        ## Store dataset
        if store_dataset:
            train_set = DatasetWrapper(train_set, store_dataset=True)
            valid_set = DatasetWrapper(valid_set, store_dataset=True)
            test_set = DatasetWrapper(test_set, store_dataset=True)

        ## set datasets
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set


class CelebASRFlowDataModule(object):
    mean = 0.5
    # std = 0.2
    std = 0.5

    def __init__(self, data_folder, scale=8, n_valid=256, rand_valid=True, store_dataset=False):
        super().__init__()

        self.img_size = 160
        self.shape = (3, self.img_size, self.img_size)

        data_folder = find_data_folder(data_folder, 'CelebA_SRFlow')
        print(f'Loading data from: {data_folder}')

        ## Base dataset
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            ])
        filenames = np.sort(os.listdir(os.path.join(data_folder, 'GT')))
        hr_filenames = [os.path.join(data_folder, 'GT', filename) for filename in filenames]
        lr_filenames = [os.path.join(data_folder, f'x{scale}', filename) for filename in filenames]
        train_set = PairsDataset(
            ImageFilesDataset(hr_filenames, transform=transform),
            ImageFilesDataset(lr_filenames, transform=transform)
        )

        ## Store dataset
        if store_dataset:
            train_set = DatasetWrapper(train_set, store_dataset=True, fixed_size_tensor=False)

        ## Split
        if n_valid != 0:
            train_set, valid_set = split_dataset(train_set, n_valid, rand=rand_valid)
        else:
            valid_set = train_set

        ## set datasets
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = valid_set

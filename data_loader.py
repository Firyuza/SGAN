from config_SGAN import cfg

import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

def get_train_valid_loader(data_dir,
                           dataset_type,
                           train_batch_size,
                           valid_batch_size,
                           augment,
                           random_seed,
                           valid_size=0.2,
                           shuffle=True,
                           show_sample=False,
                           num_workers=4,
                           pin_memory=False):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transforms
    if augment:
        if dataset_type == 'celebA':
            train_transform = transforms.Compose([
                transforms.Resize(cfg.data_augmentation.image_size),
                transforms.CenterCrop(cfg.data_augmentation.image_size),
                transforms.ToTensor(),
                normalize,
            ])
            valid_transform = transforms.Compose([
                transforms.Resize(cfg.data_augmentation.image_size),
                transforms.ToTensor(),
                normalize,
            ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    # load the dataset
    if dataset_type == 'mnist':
        train_dataset = torchvision.datasets.MNIST(
            root=data_dir, train=True, transform=train_transform, download=True)

        valid_dataset = torchvision.datasets.MNIST(
            root=data_dir, train=True, transform=train_transform, download=True)
    else:
        train_dataset = torchvision.datasets.ImageFolder(
                    root=data_dir,
                    transform=train_transform)

        valid_dataset = torchvision.datasets.ImageFolder(
                    root=data_dir,
                    transform=valid_transform)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=valid_batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    # # visualize some images
    # if show_sample:
    #     sample_loader = torch.utils.data.DataLoader(
    #         train_dataset, batch_size=9, shuffle=shuffle,
    #         num_workers=num_workers, pin_memory=pin_memory,
    #     )
    #     data_iter = iter(sample_loader)
    #     images, labels = data_iter.next()
    #     X = images.numpy().transpose([0, 2, 3, 1])
    #     plot_images(X, labels)

    return (train_loader, valid_loader)

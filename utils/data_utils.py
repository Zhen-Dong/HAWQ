from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch


class UniformDataset(Dataset):
    """
    get random uniform samples with mean 0 and variance 1
    """
    def __init__(self, length, size, transform):
        self.length = length
        self.transform = transform
        self.size = size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # var[U(-128, 127)] = (127 - (-128))**2 / 12 = 5418.75
        sample = (torch.randint(high=255, size=self.size).float() - 127.5) / 5418.75
        return sample


def getRandomData(dataset='imagenet', batch_size=512, for_inception=False):
    """
    get random sample dataloader 
    dataset: name of the dataset 
    batch_size: the batch size of random data
    for_inception: whether the data is for Inception because inception has input size 299 rather than 224
    """
    if dataset == 'cifar10':
        size = (3, 32, 32)
        num_data = 10000
    elif dataset == 'imagenet':
        num_data = 10000
        if not for_inception:
            size = (3, 224, 224)
        else:
            size = (3, 299, 299)
    else:
        raise NotImplementedError
    dataset = UniformDataset(length=num_data, size=size, transform=None)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=32)
    return data_loader


def getTestData(dataset='imagenet',
                batch_size=1024,
                path='data/imagenet',
                for_inception=False):
    """
    Get dataloader of testset 
    dataset: name of the dataset 
    batch_size: the batch size of random data
    path: the path to the data
    for_inception: whether the data is for Inception because inception has input size 299 rather than 224
    """
    if dataset == 'imagenet':
        input_size = 299 if for_inception else 224
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        test_dataset = datasets.ImageFolder(
            path + 'val',
            transforms.Compose([
                transforms.Resize(int(input_size / 0.875)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize,
            ]))

        test_loader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=32)

        return test_loader
    elif dataset == 'cifar10':
        data_dir = '/rscratch/yaohuic/data/'
        normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                         std=(0.2023, 0.1994, 0.2010))
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        test_dataset = datasets.CIFAR10(root=data_dir,
                                        train=False,
                                        transform=transform_test)
        test_loader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=32)
        return test_loader


def getTrainData(dataset='imagenet',
                batch_size=512,
                path='data/imagenet',
                for_inception=False,
                data_percentage=0.1):
    """
    Get dataloader of training
    dataset: name of the dataset
    batch_size: the batch size of random data
    path: the path to the data
    for_inception: whether the data is for Inception because inception has input size 299 rather than 224
    """
    if dataset == 'imagenet':
        input_size = 299 if for_inception else 224
        traindir = path + 'train'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        dataset_length = int(len(train_dataset) * data_percentage)
        partial_train_dataset, _ = torch.utils.data.random_split(train_dataset, [dataset_length, len(train_dataset)-dataset_length])

        train_loader = torch.utils.data.DataLoader(
            partial_train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=32, pin_memory=True)

        return train_loader

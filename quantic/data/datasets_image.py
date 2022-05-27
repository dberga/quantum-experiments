from .dataset import Dataset
from .datasetloader import DatasetLoader

import numpy as np
from torchvision import datasets, transforms


def _load_torchvision_datasubset(dataset, classes, num_classes, specific_classes_names=[],num_samples_class=-1):
    """ Utility to load a subset of a torchvision dataset
    
    Args:
        dataset (torchvision.dataset): dataset from where to extract a subset
        classes (list[str]): class names of the dataset
        num_classes (int): number of classes to load
        specific_classes_name (list[str]): if specified, load only specific class names specified in this list
        num_samples_class (int): if specified, load a maximum number of samples per class
    
    Returns:
        tuple[torchvision.dataset,list[int],list[str]): created subset dataset, list of numeric class ids, list of loaded class names
    """
    use_specific_classes = len(specific_classes_names) > 0

    dict_classes = { class_name : i for i,class_name in enumerate(classes)}
    if not use_specific_classes:
        specific_classes = range(0, num_classes)
        specific_classes_names = list(map(str,specific_classes))
    else:
        specific_classes = [dict_classes[i] for i in specific_classes_names]

    dataset.targets=np.int64(dataset.targets)

    idx = np.int64([])    
    for i in range(0,num_classes):
        class_idx = specific_classes[i]
        if num_samples_class > -1:
            idx = np.append(idx, np.where(dataset.targets == class_idx)[0][:num_samples_class])
        else:
            idx = np.append(idx, np.where(dataset.targets == class_idx)[0])

    dataset.data = dataset.data[idx]
    dataset.targets = dataset.targets[idx]

    return dataset, specific_classes, specific_classes_names


class DatasetMNIST(Dataset):
    """ Digit recognition dataset
    http://yann.lecun.com/exdb/mnist/"""

    def __init__(self,classes=["0","1","2","3","4","5","6","7","8","9"],num_classes=2,specific_classes=[],num_samples_class_train=-1,num_samples_class_test=-1,*kargs,**kwargs):
        Dataset.__init__(self,*kargs,**kwargs)

        self.classes = classes

        if self.framework == 'torchvision':

            train_dataset = datasets.MNIST(
                root="./data", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])
            )
            test_dataset = datasets.MNIST(
                root="./data", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()])
            )
        else:
            raise Exception(f'Framework {self.framework} not supported for dataset MNIST')

        # Add dataset partitions
        self.partitions['train'], self.specific_classes, self.specific_classes_names = _load_torchvision_datasubset(train_dataset, classes, num_classes, specific_classes,num_samples_class_train)
        self.partitions['test'], self.specific_classes, self.specific_classes_names =  _load_torchvision_datasubset(test_dataset, classes, num_classes, specific_classes,num_samples_class_test)

class DatasetCIFAR10(Dataset):
    """ https://www.cs.toronto.edu/~kriz/cifar.html"""

    def __init__(self,classes=["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],num_classes=2,specific_classes={},num_samples_class_train=-1,num_samples_class_test=-1,*kargs,**kwargs):
        Dataset.__init__(self,*kargs,**kwargs)

        self.classes = classes

        if self.framework == 'torchvision':

            train_dataset = datasets.CIFAR10(
                root="./data", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])
            )
            test_dataset = datasets.CIFAR10(
                root="./data", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()])
            )
        else:
            raise Exception(f'Framework {self.framework} not supported for dataset MNIST')

        # Add dataset partitions
        self.partitions['train'], self.specific_classes, self.specific_classes_names = _load_torchvision_datasubset(train_dataset, classes, num_classes, specific_classes,num_samples_class_train)
        self.partitions['test'], self.specific_classes, self.specific_classes_names =  _load_torchvision_datasubset(test_dataset, classes, num_classes, specific_classes,num_samples_class_test)

class DatasetCIFAR100(Dataset):
    """ https://www.cs.toronto.edu/~kriz/cifar.html"""

    def __init__(self,classes=['apple', 'fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'],num_classes=2,specific_classes={},num_samples_class_train=-1,num_samples_class_test=-1,*kargs,**kwargs):
        Dataset.__init__(self,*kargs,**kwargs)

        self.classes = classes

        if self.framework == 'torchvision':

            train_dataset = datasets.CIFAR100(
                root="./data", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])
            )
            test_dataset = datasets.CIFAR100(
                root="./data", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()])
            )
        else:
            raise Exception(f'Framework {self.framework} not supported for dataset MNIST')

        # Add dataset partitions
        self.partitions['train'], self.specific_classes, self.specific_classes_names = _load_torchvision_datasubset(train_dataset, classes, num_classes, specific_classes,num_samples_class_train)
        self.partitions['test'], self.specific_classes, self.specific_classes_names =  _load_torchvision_datasubset(test_dataset, classes, num_classes, specific_classes,num_samples_class_test)

DatasetLoader.register('MNIST',DatasetMNIST)
DatasetLoader.register('CIFAR10',DatasetCIFAR10)
DatasetLoader.register('CIFAR100',DatasetCIFAR100)

# TODO: DatasetCIFAR100, DatasetCIFAR10, DatasetMNIST all share same structure. Refactor into a intermediate class DatasetTorchvision
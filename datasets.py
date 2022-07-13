import os
import torch
import torch.nn.parallel
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import numpy as np
from PIL import Image
import copy
from utils import AverageAccumulator, VectorAccumulator, accuracy, Progressbar, adjust_learning_rate, get_num_parameters

class CIFAR_Multi_Task(data.Dataset):
    def __init__(self, CIFAR_obj, task_dict, sample_percent=100):

        task_list = []
        task_id_list = []
        for td in task_dict:
            task_list.append(td['class_id'])
            task_id_list.append(td['tid'])

        class_idx = []
        for task in task_list:
            class_idx.extend(task)

        self.transform = CIFAR_obj.transform
        self.target_transform = CIFAR_obj.target_transform
        self.train = CIFAR_obj.train
        self.task_list = task_list
        self.org_class_names, self.remapped_class_names = self.get_class_names(CIFAR_obj.class_to_idx, task_list)
        self.sample_percent = sample_percent

        # Create sub set of data from classes

        # labels = CIFAR_obj.train_labels if self.train else CIFAR_obj.test_labels
        # data = CIFAR_obj.train_data if self.train else CIFAR_obj.test_data

        targets = CIFAR_obj.targets
        data = CIFAR_obj.data

        target_idx = self.get_match_index(targets, class_idx)

        data = data[target_idx, :, :, :]
        temp = np.array([targets])
        targets = temp[0][target_idx].tolist()

        # now load the picked numpy arrays
        self.remapped_targets = self.remap_targets(targets, task_list)
        self.targets = targets
        self.data = data

        if self.sample_percent is not 100:
            self.data, self.targets, self.remapped_targets = self.sample_data()
        # print('asd')

        self.task_id = []
        for t in self.targets:
            for j,tl in enumerate(task_list):
                if t in tl:
                    self.task_id.append(task_id_list[j])

        # print('asd')

    def sample_data(self):
        unique_targets = np.unique(self.targets)
        unique_counts = np.zeros(len(unique_targets))
        for t in self.targets:
            idx = np.where(unique_targets == t)
            unique_counts[idx] += 1

        unique_counts = np.int16(np.ceil(unique_counts*(self.sample_percent/100)))

        targets = []
        remapped_targets = []
        data = np.zeros([unique_counts.sum(), 32, 32, 3], 'uint8')

        counts = np.zeros(len(unique_targets))

        c = 0
        for i, t in enumerate(self.targets):
            idx = np.where(unique_targets == t)
            if counts[idx] < unique_counts[idx]:

                targets.append(self.targets[i])
                remapped_targets.append(self.remapped_targets[i])
                data[c, :, :, :] = self.data[i, :, :, :]

                c += 1
                counts[idx] += 1

        return data, targets, remapped_targets


    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, remapped_target = self.data[index], self.targets[index], self.remapped_targets[index]
        task_id = self.task_id[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, remapped_target, task_id

    def get_match_index(self, targets, class_idx):
        target_indices = []
        # new_targets = []

        for i in range(len(targets)):
            if targets[i] in class_idx:
                target_indices.append(i)
                # new_targets.append(np.where(np.array(class_idx) == targets[i])[0].item())

        return target_indices

    def remap_targets(self, targets, task_list):
        remapped_targets = []

        targets = np.array(targets)

        for i, t in enumerate(targets):
            for j, tl in enumerate(task_list):
                if t in tl:
                    temp = np.where(tl == t)
                    remapped_targets.append(temp[0][0])


        # unique_targets = np.unique(targets)
        # for t in targets:
        #     remapped_targets.append(np.where(task_list == t)[0][0])

        return remapped_targets

    def get_class_names(self, class_to_idx, task_list):

        org_class_names = [{} for i in range(len(task_list))]
        remapped_class_names = [{} for i in range(len(task_list))]

        keys = list(class_to_idx.keys())
        values = list(class_to_idx.values())

        for i, tl in enumerate(task_list):
            for j, cid in enumerate(tl):
                if cid in values:
                    org_class_names[i].update({keys[cid]:values[cid]})
                    remapped_class_names[i].update({keys[cid]:j})

        return org_class_names, remapped_class_names

def get_cifar_data(data_set, split, batch_size=100, num_workers=4):
    print('==> Preparing dataset %s' % data_set)
    # transform_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225]),
    # ])
    #
    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225]),
    # ])

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if data_set in ['CIFAR10', 'cifar10']:
        dataloader = datasets.CIFAR10
        num_classes = 10
        trainset = dataloader(root=os.path.join('.','../../data/CIFAR'), train=True, download=True, transform=transform_train)
        testset = dataloader(root=os.path.join('.','../../data/CIFAR'), train=False, download=False, transform=transform_test)

    elif data_set in ['CIFAR100', 'cifar100']:
        dataloader = datasets.CIFAR100
        num_classes = 100
        trainset = dataloader(root=os.path.join('.','../../data/CIFAR'), train=True, download=True, transform=transform_train)
        testset = dataloader(root=os.path.join('.','../../data/CIFAR'), train=False, download=False, transform=transform_test)

    if split == 'train':
        trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        return trainset, trainloader, num_classes
    elif split == 'test':
        testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return testset, testloader, num_classes

def get_cifar_multitask(dataset, task_dict, split, batch_size=100, num_workers=4, sample_percent=100):

    if not isinstance(task_dict, list):
        task_dict = [task_dict]

    temp_set, _, _ = get_cifar_data(dataset, split, batch_size, num_workers)
    dset = CIFAR_Multi_Task(temp_set, task_dict, sample_percent)

    if split == 'train':
        dloader = data.DataLoader(dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    elif split == 'test':
        dloader = data.DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return dset, dloader


# data, _ = get_cifar_multitask('cifar10', [[1,2], [3, 4, 5,6,7], [8,9,10]], 'train', batch_size=100, num_workers=4, sample_percent=1)
# data[1000]

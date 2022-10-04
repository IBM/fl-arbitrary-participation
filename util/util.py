import numpy as np
import math
from config import mixing_ratio
from torch.utils.data import Dataset


class NodeSampler:
    def __init__(self, n_nodes, permutation=True):
        self.n_nodes = n_nodes
        self.permutation = permutation
        self.remaining_permutation = []

    def sample(self, node_sample_set, size):
        if self.permutation:
            sampled_set = []
            while len(sampled_set) < size:
                if len(self.remaining_permutation) == 0:
                    self.remaining_permutation = list(np.random.permutation(self.n_nodes))

                i = self.remaining_permutation.pop()

                if i in node_sample_set:
                    sampled_set.append(i)
        else:
            sampled_set = np.random.choice(node_sample_set, size, replace=False)

        return sampled_set


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def mnist_partition(dataset, n_nodes):
    # TODO Check whether MNIST and CIFAR can be combined into the same function

    dict_users = {i: np.array([], dtype='int64') for i in range(n_nodes)}
    labels = dataset.targets.numpy()

    min_label = min(labels)
    max_label = max(labels)
    num_labels = max_label - min_label + 1

    if n_nodes > num_labels:
        label_for_nodes = []
        for n in range(0, n_nodes):
            for i in range(0, num_labels):
                if int(np.round(i * n_nodes / num_labels)) <= n < int(np.round((i + 1) * n_nodes / num_labels)):
                    label_for_nodes.append(i + min_label)
                    
    for i in range(0, len(labels)):
        if np.random.rand() <= mixing_ratio:
            tmp_target_node = np.random.randint(n_nodes)
        else:
            tmp_target_node = int((labels[i] - min_label) % n_nodes)
            if n_nodes > num_labels:
                tmp_min_index = 0
                tmp_min_val = math.inf
                for n in range(0, n_nodes):
                    if label_for_nodes[n] == labels[i] and len(dict_users[n]) < tmp_min_val:
                        tmp_min_val = len(dict_users[n])
                        tmp_min_index = n
                tmp_target_node = tmp_min_index
        dict_users[tmp_target_node] = np.concatenate((dict_users[tmp_target_node], [i]), axis=0)

    return dict_users


def cifar_partition(dataset, n_nodes):

    dict_users = {i: np.array([], dtype='int64') for i in range(n_nodes)}
    labels = np.array(dataset.targets)

    min_label = min(labels)
    max_label = max(labels)
    num_labels = max_label - min_label + 1

    if n_nodes > num_labels:
        label_for_nodes = []
        for n in range(0, n_nodes):
            for i in range(0, num_labels):
                if int(np.round(i * n_nodes / num_labels)) <= n < int(np.round((i + 1) * n_nodes / num_labels)):
                    label_for_nodes.append(i + min_label)

    for i in range(0, len(labels)):
        if np.random.rand() <= mixing_ratio:
            tmp_target_node = np.random.randint(n_nodes)
        else:
            tmp_target_node = int((labels[i] - min_label) % n_nodes)
            if n_nodes > num_labels:
                tmp_min_index = 0
                tmp_min_val = math.inf
                for n in range(0, n_nodes):
                    if label_for_nodes[n] == labels[i] and len(dict_users[n]) < tmp_min_val:
                        tmp_min_val = len(dict_users[n])
                        tmp_min_index = n
                tmp_target_node = tmp_min_index
        dict_users[tmp_target_node] = np.concatenate((dict_users[tmp_target_node], [i]), axis=0)

    return dict_users


def split_data(dataset, data_train, n_nodes):
    if dataset == 'FashionMNIST':
        dict_users = mnist_partition(data_train, n_nodes)
    elif dataset == 'CIFAR10':
        dict_users = cifar_partition(data_train, n_nodes)
    else:
        raise Exception('Unknown dataset name.')
    return dict_users



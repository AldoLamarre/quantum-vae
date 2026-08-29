import torch
from torch.utils.data import Dataset


class LabelMappedDataset(Dataset):
    def __init__(self, samples, label_map):
        self.samples = list(samples)
        self.label_map = dict(label_map)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, label = self.samples[idx]
        return image, self.label_map.get(label, label)


def filter_dataset_by_labels(dataset, allowed_labels):
    allowed_labels = set(allowed_labels)
    return [(data, target) for data, target in dataset if target in allowed_labels]


def split_train_val(dataset, train_ratio):
    train_len = int(len(dataset) * train_ratio)
    val_len = len(dataset) - train_len
    return torch.utils.data.random_split(dataset, [train_len, val_len])


def latent_outer_product(state):
    return torch.einsum("bi,bj->bij", state, state).reshape(state.shape[0], -1)


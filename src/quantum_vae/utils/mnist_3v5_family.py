from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from .mnist_dataset_utils import build_label_mapped_splits


LABEL_MAP = {3: 0, 5: 1}


def build_mnist_3v5_dataset_bundle(
    training_data,
    test_data,
    batch_size=128,
    train_ratio=0.9,
    filtered_test_source="train",
):
    filtered_train, filtered_test, train_set, val_set = build_label_mapped_splits(
        training_data,
        test_data,
        LABEL_MAP,
        train_ratio=train_ratio,
        filtered_test_source=filtered_test_source,
    )

    return {
        "label_map": LABEL_MAP,
        "filtered_train": filtered_train,
        "filtered_test": filtered_test,
        "train_set": train_set,
        "val_set": val_set,
        "train_dataloader": DataLoader(train_set, batch_size=batch_size),
        "val_dataloader": DataLoader(val_set, batch_size=batch_size),
        "test_dataloader": DataLoader(filtered_test, batch_size=batch_size),
    }


def load_mnist_3v5_dataset_bundle(
    root="data",
    batch_size=128,
    train_ratio=0.9,
    filtered_test_source="train",
    download=False,
):
    training_data = datasets.MNIST(
        root=root,
        train=True,
        download=download,
        transform=ToTensor(),
    )
    test_data = datasets.MNIST(
        root=root,
        train=False,
        download=download,
        transform=ToTensor(),
    )

    return build_mnist_3v5_dataset_bundle(
        training_data,
        test_data,
        batch_size=batch_size,
        train_ratio=train_ratio,
        filtered_test_source=filtered_test_source,
    )

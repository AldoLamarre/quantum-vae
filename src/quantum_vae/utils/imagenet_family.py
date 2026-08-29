from torch.utils.data import DataLoader


def build_imagenet_data_bundle(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size=128,
):
    """Build ImageNet data bundle with pre-split datasets.
    
    ImageNet is loaded as separate train/val/test splits from HuggingFace,
    so we don't use random_split like CIFAR. This helper wraps them into
    DataLoaders with consistent interface.
    """
    return {
        "train_set": train_dataset,
        "val_set": val_dataset,
        "test_set": test_dataset,
        "train_dataloader": DataLoader(train_dataset, batch_size=batch_size),
        "val_dataloader": DataLoader(val_dataset, batch_size=batch_size),
        "test_dataloader": DataLoader(test_dataset, batch_size=batch_size),
    }

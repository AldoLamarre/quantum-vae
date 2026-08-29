from torch.utils.data import DataLoader, random_split


def build_cifar10_data_bundle(
    training_data,
    test_data,
    batch_size=128,
    train_size=40000,
    val_size=10000,
):
    train_set, val_set = random_split(training_data, [train_size, val_size])
    return {
        "train_set": train_set,
        "val_set": val_set,
        "test_set": test_data,
        "train_dataloader": DataLoader(train_set, batch_size=batch_size),
        "val_dataloader": DataLoader(val_set, batch_size=batch_size),
        "test_dataloader": DataLoader(test_data, batch_size=batch_size),
    }

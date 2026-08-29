from torch.utils.data import DataLoader

from .mnist_dataset_utils import build_label_mapped_splits
from .mnist_family import build_mnist_data_bundle
from .script_utils import latent_outer_product


def build_mnist_legacy_ablation_bundle(
    training_data,
    test_data,
    batch_size=128,
    label_map=None,
    train_size=50000,
    val_size=10000,
    train_ratio=0.9,
    filtered_test_source="train",
):
    """Build the MNIST bundle used by the legacy MLP-VAE ablation family.

    If label_map is provided, the bundle is filtered to the selected labels.
    Otherwise the full MNIST split is returned.
    """
    if label_map is None:
        return build_mnist_data_bundle(
            training_data,
            test_data,
            batch_size=batch_size,
            train_size=train_size,
            val_size=val_size,
        )

    filtered_train, filtered_test, train_set, val_set = build_label_mapped_splits(
        training_data,
        test_data,
        label_map,
        train_ratio=train_ratio,
        filtered_test_source=filtered_test_source,
    )

    return {
        "label_map": label_map,
        "filtered_train": filtered_train,
        "filtered_test": filtered_test,
        "train_set": train_set,
        "val_set": val_set,
        "train_dataloader": DataLoader(train_set, batch_size=batch_size),
        "val_dataloader": DataLoader(val_set, batch_size=batch_size),
        "test_dataloader": DataLoader(filtered_test, batch_size=batch_size),
    }


def legacy_amplitude_features(state):
    return state.real, state.imag


def legacy_outer_product_features(state):
    return latent_outer_product(state)


def legacy_pixels_features(x):
    return x.view(x.size(0), -1)

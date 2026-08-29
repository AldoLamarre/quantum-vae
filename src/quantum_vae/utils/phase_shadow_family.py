from .mnist_family import build_mnist_data_bundle
from .runtime_utils import resolve_device


def build_phase_shadow_data_bundle(
    training_data,
    test_data,
    batch_size=128,
    train_size=50000,
    val_size=10000,
):
    return build_mnist_data_bundle(
        training_data,
        test_data,
        batch_size=batch_size,
        train_size=train_size,
        val_size=val_size,
    )


def resolve_phase_shadow_device():
    return resolve_device()

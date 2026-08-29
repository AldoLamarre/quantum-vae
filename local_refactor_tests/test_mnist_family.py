import unittest

from src.quantum_vae.utils.mnist_family import build_mnist_data_bundle


class MnistFamilyTests(unittest.TestCase):
    def test_build_mnist_data_bundle_preserves_standard_split(self):
        training_data = list(range(60000))
        test_data = list(range(60000, 70000))

        bundle = build_mnist_data_bundle(
            training_data,
            test_data,
            batch_size=2,
            train_size=50000,
            val_size=10000,
        )

        self.assertEqual(len(bundle["train_set"]), 50000)
        self.assertEqual(len(bundle["val_set"]), 10000)
        self.assertEqual(len(bundle["test_dataloader"]), 5000)
        self.assertEqual(len(bundle["train_dataloader"]), 25000)
        self.assertEqual(len(bundle["val_dataloader"]), 5000)


if __name__ == "__main__":
    unittest.main()

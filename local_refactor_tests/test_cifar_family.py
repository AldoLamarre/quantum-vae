import unittest

from src.quantum_vae.utils.cifar_family import build_cifar10_data_bundle


class CifarFamilyTests(unittest.TestCase):
    def test_build_cifar10_data_bundle_preserves_standard_split(self):
        training_data = list(range(50000))
        test_data = list(range(50000, 60000))

        bundle = build_cifar10_data_bundle(
            training_data,
            test_data,
            batch_size=2,
            train_size=40000,
            val_size=10000,
        )

        self.assertEqual(len(bundle["train_set"]), 40000)
        self.assertEqual(len(bundle["val_set"]), 10000)
        self.assertEqual(len(bundle["train_dataloader"]), 20000)
        self.assertEqual(len(bundle["val_dataloader"]), 5000)
        self.assertEqual(len(bundle["test_dataloader"]), 5000)


if __name__ == "__main__":
    unittest.main()

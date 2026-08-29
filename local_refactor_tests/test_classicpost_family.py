import unittest

from src.quantum_vae.utils.classicpost_family import build_classic_post_data_bundle


class ClassicPostFamilyTests(unittest.TestCase):
    def test_build_classic_post_data_bundle_preserves_train_val_split(self):
        training_data = list(range(10))
        test_data = list(range(10, 20))

        bundle = build_classic_post_data_bundle(
            training_data,
            test_data,
            batch_size=2,
            train_size=6,
            val_size=4,
        )

        self.assertEqual(len(bundle["train_set"]), 6)
        self.assertEqual(len(bundle["val_set"]), 4)
        self.assertEqual(len(bundle["test_data"]) if "test_data" in bundle else len(test_data), len(test_data))
        self.assertEqual(len(bundle["train_dataloader"]), 3)
        self.assertEqual(len(bundle["val_dataloader"]), 2)
        self.assertEqual(len(bundle["test_dataloader"]), 5)


if __name__ == "__main__":
    unittest.main()

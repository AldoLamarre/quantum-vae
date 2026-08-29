import unittest

from src.quantum_vae.utils.mnist_3v5_family import build_mnist_3v5_dataset_bundle


class Mnist3v5FamilyTests(unittest.TestCase):
    def test_build_bundle_keeps_3_vs_5_label_mapping(self):
        training_data = [("train3", 3), ("train4", 4), ("train5", 5)]
        test_data = [("test3", 3), ("test5", 5)]

        bundle = build_mnist_3v5_dataset_bundle(
            training_data,
            test_data,
            batch_size=2,
            train_ratio=0.5,
            filtered_test_source="train",
        )

        self.assertEqual(bundle["label_map"], {3: 0, 5: 1})
        self.assertEqual(len(bundle["filtered_train"]), 2)
        self.assertEqual(bundle["filtered_test"][0][0], "train3")
        self.assertEqual(bundle["filtered_test"][0][1], 0)
        self.assertEqual(len(bundle["train_set"]) + len(bundle["val_set"]), 2)
        self.assertEqual(len(bundle["train_dataloader"]), 1)


if __name__ == "__main__":
    unittest.main()

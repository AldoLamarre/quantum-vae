import unittest

from src.quantum_vae.utils.mnist_dataset_utils import build_label_mapped_splits


class MnistDatasetUtilsTests(unittest.TestCase):
    def test_build_label_mapped_splits_can_reuse_training_data_for_filtered_test(self):
        training_data = [("train3", 3), ("train4", 4), ("train5", 5)]
        test_data = [("test3", 3), ("test5", 5)]

        filtered_train, filtered_test, train_set, val_set = build_label_mapped_splits(
            training_data,
            test_data,
            {3: 0, 5: 1},
            train_ratio=0.5,
            filtered_test_source="train",
        )

        self.assertEqual(len(filtered_train), 2)
        self.assertEqual(filtered_test[0], ("train3", 0))
        self.assertEqual(filtered_test[1], ("train5", 1))
        self.assertEqual(len(train_set) + len(val_set), 2)

    def test_build_label_mapped_splits_can_use_real_test_data(self):
        training_data = [("train3", 3), ("train4", 4), ("train5", 5)]
        test_data = [("test3", 3), ("test5", 5)]

        _, filtered_test, _, _ = build_label_mapped_splits(
            training_data,
            test_data,
            {3: 0, 5: 1},
            filtered_test_source="test",
        )

        self.assertEqual(filtered_test[0], ("test3", 0))
        self.assertEqual(filtered_test[1], ("test5", 1))


if __name__ == "__main__":
    unittest.main()

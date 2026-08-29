import unittest

import torch

from src.quantum_vae.utils.script_utils import (
    LabelMappedDataset as PackageLabelMappedDataset,
    filter_dataset_by_labels as package_filter_dataset_by_labels,
)
from script_utils import LabelMappedDataset, filter_dataset_by_labels, latent_outer_product, split_train_val


class ScriptUtilsTests(unittest.TestCase):
    def test_filter_dataset_by_labels_keeps_only_requested_labels(self):
        dataset = [("a", 3), ("b", 2), ("c", 5), ("d", 8)]
        filtered = filter_dataset_by_labels(dataset, {3, 5})
        self.assertEqual(filtered, [("a", 3), ("c", 5)])

    def test_label_mapped_dataset_remaps_labels(self):
        dataset = LabelMappedDataset([("img3", 3), ("img5", 5)], {3: 0, 5: 1})
        self.assertEqual(dataset[0], ("img3", 0))
        self.assertEqual(dataset[1], ("img5", 1))

    def test_root_compatibility_module_matches_packaged_script_utils(self):
        dataset = [("a", 3), ("b", 2), ("c", 5)]
        self.assertEqual(
            filter_dataset_by_labels(dataset, {3, 5}),
            package_filter_dataset_by_labels(dataset, {3, 5}),
        )

        packaged_dataset = PackageLabelMappedDataset([("img3", 3)], {3: 0})
        compatibility_dataset = LabelMappedDataset([("img3", 3)], {3: 0})
        self.assertEqual(compatibility_dataset[0], packaged_dataset[0])

    def test_split_train_val_preserves_total_size(self):
        dataset = list(range(10))
        train_set, val_set = split_train_val(dataset, 0.9)
        self.assertEqual(len(train_set), 9)
        self.assertEqual(len(val_set), 1)
        self.assertEqual(len(train_set) + len(val_set), 10)

    def test_latent_outer_product_matches_original_expression(self):
        state = torch.tensor([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]])
        expected = torch.einsum("bi,bj->bij", state, state).view(-1, state.shape[-1] * state.shape[-1])
        actual = latent_outer_product(state)
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()

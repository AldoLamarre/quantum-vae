import unittest

from src.quantum_vae.utils.phase_shadow_family import (
    build_phase_shadow_data_bundle,
    resolve_phase_shadow_device,
)
from src.quantum_vae.utils.runtime_utils import resolve_device


class PhaseShadowFamilyTests(unittest.TestCase):
    def test_build_phase_shadow_data_bundle_preserves_standard_split(self):
        training_data = list(range(60000))
        test_data = list(range(60000, 70000))

        bundle = build_phase_shadow_data_bundle(
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

    def test_resolve_phase_shadow_device_matches_runtime_utils(self):
        self.assertEqual(resolve_phase_shadow_device(), resolve_device())


if __name__ == "__main__":
    unittest.main()

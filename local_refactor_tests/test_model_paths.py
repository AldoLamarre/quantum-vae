import tempfile
import unittest
from pathlib import Path

from src.quantum_vae.utils.model_paths import (
    registered_model_path as packaged_registered_model_path,
)
from model_paths import (
    checkpoints_dir,
    managed_model_path,
    models_dir,
    registered_model_path,
)


class ModelPathsTests(unittest.TestCase):
    def test_registered_model_path_uses_legacy_file_when_managed_file_is_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            legacy_file = Path(tmpdir) / "variationalautoencodertestpennylane five.pt"
            legacy_file.touch()

            resolved = registered_model_path("mnist_pennylane_vae_five", project_root=tmpdir)

            self.assertEqual(resolved, str(legacy_file))

    def test_registered_model_path_prefers_managed_file_when_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            legacy_file = Path(tmpdir) / "variationalautoencodertestpennylane five.pt"
            managed_file = Path(tmpdir) / "models/autoencoders/mnist/variationalautoencodertestpennylane_five.pt"
            legacy_file.touch()
            managed_file.parent.mkdir(parents=True, exist_ok=True)
            managed_file.touch()

            resolved = registered_model_path("mnist_pennylane_vae_five", project_root=tmpdir)

            self.assertEqual(resolved, str(managed_file))

    def test_managed_model_path_can_create_parent_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            managed_path = managed_model_path(
                "cifar10_autoencoderkl",
                project_root=tmpdir,
                create_parent=True,
            )

            self.assertEqual(
                managed_path,
                str(Path(tmpdir) / "models/autoencoders/cifar10/autoencoderklcifar10.pt"),
            )
            self.assertTrue((Path(tmpdir) / "models/autoencoders/cifar10").is_dir())

    def test_models_and_checkpoints_dir_can_be_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertEqual(models_dir(tmpdir, create=True), Path(tmpdir) / "models")
            self.assertEqual(checkpoints_dir(tmpdir, create=True), Path(tmpdir) / "checkpoints")
            self.assertTrue((Path(tmpdir) / "models").is_dir())
            self.assertTrue((Path(tmpdir) / "checkpoints").is_dir())

    def test_root_model_paths_compatibility_matches_packaged_model_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            legacy_file = Path(tmpdir) / "variationalautoencodertestpennylane five.pt"
            legacy_file.touch()

            compatibility_resolved = registered_model_path("mnist_pennylane_vae_five", project_root=tmpdir)
            packaged_resolved = packaged_registered_model_path("mnist_pennylane_vae_five", project_root=tmpdir)

            self.assertEqual(compatibility_resolved, packaged_resolved)


if __name__ == "__main__":
    unittest.main()

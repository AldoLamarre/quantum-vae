import os
import tempfile
import unittest

from src.quantum_vae.utils.runtime_utils import (
    create_run_path as package_create_run_path,
)
from runtime_utils import configure_cuda_visible_devices, create_run_path, resolve_device


class RuntimeUtilsTests(unittest.TestCase):
    def test_create_run_path_without_timestamp_uses_base_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = create_run_path(tmpdir, include_timestamp=False)
            self.assertEqual(path, f"{tmpdir}/")
            self.assertTrue(os.path.isdir(tmpdir))

    def test_create_run_path_with_explicit_timestamp_creates_subdirectory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = create_run_path(tmpdir, timestamp="fixed-stamp")
            self.assertEqual(path, f"{tmpdir}/fixed-stamp/")
            self.assertTrue(os.path.isdir(os.path.join(tmpdir, "fixed-stamp")))

    def test_root_runtime_compatibility_matches_packaged_runtime_utils(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            compatibility_path = create_run_path(tmpdir, timestamp="compat", include_timestamp=True)
            packaged_path = package_create_run_path(tmpdir, timestamp="package", include_timestamp=True)
            self.assertEqual(compatibility_path, f"{tmpdir}/compat/")
            self.assertEqual(packaged_path, f"{tmpdir}/package/")

    def test_configure_cuda_visible_devices_sets_variable(self):
        previous = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            configure_cuda_visible_devices("7")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "7")
        finally:
            if previous is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous

    def test_configure_cuda_visible_devices_respects_only_if_unset(self):
        previous = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"
            configure_cuda_visible_devices("7", only_if_unset=True)
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "1")
        finally:
            if previous is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous

    def test_resolve_device_returns_known_value(self):
        self.assertIn(resolve_device(), {"cuda", "mps", "cpu"})


if __name__ == "__main__":
    unittest.main()

import unittest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from nixglhost import (
    CacheDirContent,
    LibraryPath,
    ResolvedLib,
    resolve_libraries,
    copy_and_patch_libs,
    CUDA_DSO_PATTERNS,
)


class TestCacheSerializer(unittest.TestCase):
    def test_hostdso_json_golden_test(self):
        lp = LibraryPath(
            glx=[
                ResolvedLib(
                    "dummyglx.so", "/lib", "/lib/dummyglx.so", 1670260550.481498, 1612
                )
            ],
            cuda=[
                ResolvedLib(
                    "dummycuda.so", "/lib", "/lib/dummycuda.so", 2670260550.481498, 2612
                )
            ],
            generic=[
                ResolvedLib(
                    "dummygeneric.so",
                    "/lib",
                    "/lib/dummygeneric.so",
                    3670260550.481498,
                    3612,
                )
            ],
            egl=[
                ResolvedLib(
                    "dummyegl.so", "/lib", "/lib/dummyegl.so", 4670260550.481498, 4612
                )
            ],
            path="/path/to/lib/dir",
        )
        cdc = CacheDirContent([lp])
        json = cdc.to_json()

        self.assertIsNotNone(json)
        golden_cdc = CacheDirContent.from_json(json)
        self.assertEqual(cdc, golden_cdc)
        self.assertEqual(cdc.to_json(), golden_cdc.to_json())

    def test_eq_commut_jsons(self):
        """Checks that object equality is not sensible to JSON keys commutations"""
        cwd = os.path.dirname(os.path.realpath(__file__))
        with open(
            os.path.join(cwd, "..", "tests", "fixtures", "json_permut", "1.json"),
            "r",
            encoding="utf8",
        ) as f:
            cdc_json = f.read()
        with open(
            os.path.join(cwd, "..", "tests", "fixtures", "json_permut", "2.json"),
            "r",
            encoding="utf8",
        ) as f:
            commut_cdc_json = f.read()
        with open(
            os.path.join(
                cwd, "..", "tests", "fixtures", "json_permut", "not-equal.json"
            ),
            "r",
            encoding="utf8",
        ) as f:
            wrong_cdc_json = f.read()
        cdc = CacheDirContent.from_json(cdc_json)
        commut_cdc = CacheDirContent.from_json(commut_cdc_json)
        wrong_cdc = CacheDirContent.from_json(wrong_cdc_json)
        self.assertEqual(cdc, commut_cdc)
        self.assertNotEqual(cdc, wrong_cdc)
        self.assertNotEqual(commut_cdc, wrong_cdc)


class TestSymlinkPreservation(unittest.TestCase):
    """Tests for symlink structure preservation"""

    def setUp(self):
        """Create a temporary directory with test files and symlinks"""
        self.temp_dir = tempfile.mkdtemp()
        self.source_dir = Path(self.temp_dir) / "source"
        self.dest_dir = Path(self.temp_dir) / "dest"
        self.source_dir.mkdir()
        self.dest_dir.mkdir()

    def tearDown(self):
        """Clean up temporary directory"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_resolve_libraries_detects_symlinks(self):
        """Test that resolve_libraries correctly identifies symlinks"""
        # Create a real file
        real_file = self.source_dir / "libcuda.so.565.57.01"
        real_file.write_text("fake library content")

        # Create symlink chain
        (self.source_dir / "libcuda.so.1").symlink_to("libcuda.so.565.57.01")
        (self.source_dir / "libcuda.so").symlink_to("libcuda.so.1")

        # Resolve libraries
        resolved_libs = resolve_libraries(str(self.source_dir), CUDA_DSO_PATTERNS)

        # Should find 3 items (1 real file + 2 symlinks)
        self.assertEqual(len(resolved_libs), 3)

        # Find each library type
        libs_by_name = {lib.name: lib for lib in resolved_libs}

        # Check the real file
        self.assertIn("libcuda.so.565.57.01", libs_by_name)
        self.assertFalse(libs_by_name["libcuda.so.565.57.01"].is_symlink)
        self.assertIsNone(libs_by_name["libcuda.so.565.57.01"].symlink_target)

        # Check first symlink
        self.assertIn("libcuda.so.1", libs_by_name)
        self.assertTrue(libs_by_name["libcuda.so.1"].is_symlink)
        self.assertEqual(
            libs_by_name["libcuda.so.1"].symlink_target, "libcuda.so.565.57.01"
        )

        # Check second symlink
        self.assertIn("libcuda.so", libs_by_name)
        self.assertTrue(libs_by_name["libcuda.so"].is_symlink)
        self.assertEqual(libs_by_name["libcuda.so"].symlink_target, "libcuda.so.1")

    def test_symlink_structure_preserved_after_copy(self):
        """Test that copy_and_patch_libs preserves symlink structure"""
        # Create test files
        real_file = self.source_dir / "libcuda.so.565.57.01"
        real_file.write_text("fake library content")
        (self.source_dir / "libcuda.so.1").symlink_to("libcuda.so.565.57.01")
        (self.source_dir / "libcuda.so").symlink_to("libcuda.so.1")

        # Resolve libraries
        resolved_libs = resolve_libraries(str(self.source_dir), CUDA_DSO_PATTERNS)

        # Mock patch_dsos to avoid calling patchelf
        with patch("nixglhost.patch_dsos"):
            copy_and_patch_libs(resolved_libs, str(self.dest_dir))

        # Verify all files/symlinks were copied
        self.assertTrue((self.dest_dir / "libcuda.so.565.57.01").exists())
        self.assertTrue((self.dest_dir / "libcuda.so.1").exists())
        self.assertTrue((self.dest_dir / "libcuda.so").exists())

        # Verify symlink structure
        self.assertFalse((self.dest_dir / "libcuda.so.565.57.01").is_symlink())
        self.assertTrue((self.dest_dir / "libcuda.so.1").is_symlink())
        self.assertTrue((self.dest_dir / "libcuda.so").is_symlink())

        # Verify symlink targets
        self.assertEqual(
            str((self.dest_dir / "libcuda.so.1").readlink()), "libcuda.so.565.57.01"
        )
        self.assertEqual(str((self.dest_dir / "libcuda.so").readlink()), "libcuda.so.1")

    def test_symlink_chain_resolution(self):
        """Test that complex symlink chains are correctly preserved"""
        # Create a real file
        real_file = self.source_dir / "libnvidia-ml.so.565.57.01"
        real_file.write_text("fake nvidia-ml library")

        # Create multiple levels of symlinks
        (self.source_dir / "libnvidia-ml.so.1").symlink_to("libnvidia-ml.so.565.57.01")
        (self.source_dir / "libnvidia-ml.so").symlink_to("libnvidia-ml.so.1")

        resolved_libs = resolve_libraries(str(self.source_dir), CUDA_DSO_PATTERNS)

        with patch("nixglhost.patch_dsos"):
            copy_and_patch_libs(resolved_libs, str(self.dest_dir))

        # Verify the entire chain resolves correctly
        final_target = (self.dest_dir / "libnvidia-ml.so").resolve()
        self.assertEqual(final_target.name, "libnvidia-ml.so.565.57.01")

        # Verify each link in the chain
        first_link_target = (self.dest_dir / "libnvidia-ml.so").readlink()
        self.assertEqual(str(first_link_target), "libnvidia-ml.so.1")

        second_link_target = (self.dest_dir / "libnvidia-ml.so.1").readlink()
        self.assertEqual(str(second_link_target), "libnvidia-ml.so.565.57.01")

    def test_resolved_lib_serialization_with_symlinks(self):
        """Test that ResolvedLib with symlink info serializes correctly"""
        lib_symlink = ResolvedLib(
            name="libcuda.so",
            dirpath="/usr/lib",
            fullpath="/usr/lib/libcuda.so",
            last_modification=1234567890.0,
            size=0,
            is_symlink=True,
            symlink_target="libcuda.so.1",
        )

        # Convert to dict and back
        lib_dict = lib_symlink.to_dict()
        lib_restored = ResolvedLib.from_dict(lib_dict)

        # Verify symlink info is preserved
        self.assertTrue(lib_restored.is_symlink)
        self.assertEqual(lib_restored.symlink_target, "libcuda.so.1")
        self.assertEqual(lib_symlink, lib_restored)

    def test_cache_version_with_symlinks(self):
        """Test that cache with symlinks serializes correctly"""
        lib_real = ResolvedLib(
            name="libcuda.so.565.57.01",
            dirpath="/usr/lib",
            fullpath="/usr/lib/libcuda.so.565.57.01",
            last_modification=1234567890.0,
            size=1024,
            is_symlink=False,
            symlink_target=None,
        )

        lib_symlink = ResolvedLib(
            name="libcuda.so",
            dirpath="/usr/lib",
            fullpath="/usr/lib/libcuda.so",
            last_modification=1234567890.0,
            size=0,
            is_symlink=True,
            symlink_target="libcuda.so.565.57.01",
        )

        lp = LibraryPath(
            glx=[], cuda=[lib_real, lib_symlink], generic=[], egl=[], path="/usr/lib"
        )

        cache = CacheDirContent([lp])
        json_str = cache.to_json()

        # Deserialize and verify
        cache_restored = CacheDirContent.from_json(json_str)
        self.assertEqual(cache, cache_restored)

        # Verify symlink info is in the restored cache
        restored_cuda_libs = cache_restored.paths[0].cuda
        symlink_libs = [lib for lib in restored_cuda_libs if lib.is_symlink]
        self.assertEqual(len(symlink_libs), 1)
        self.assertEqual(symlink_libs[0].symlink_target, "libcuda.so.565.57.01")

    def test_symlink_pointing_outside_directory_warns(self):
        """Test that symlinks pointing outside the directory generate warnings"""
        import sys
        from io import StringIO

        # Create a real file
        real_file = self.source_dir / "libcuda.so.565.57.01"
        real_file.write_text("fake library content")

        # Manually create a ResolvedLib with a symlink pointing outside
        lib_with_bad_symlink = ResolvedLib(
            name="libcuda.so",
            dirpath=str(self.source_dir),
            fullpath=str(self.source_dir / "libcuda.so"),
            last_modification=1234567890.0,
            size=0,
            is_symlink=True,
            symlink_target="../other/libcuda.so.565.57.01",
        )

        lib_real = ResolvedLib(
            name="libcuda.so.565.57.01",
            dirpath=str(self.source_dir),
            fullpath=str(real_file),
            is_symlink=False,
            symlink_target=None,
        )

        # Capture stderr to check for warning
        old_stderr = sys.stderr
        sys.stderr = StringIO()

        try:
            with patch("nixglhost.patch_dsos"):
                copy_and_patch_libs(
                    [lib_real, lib_with_bad_symlink], str(self.dest_dir)
                )

            stderr_output = sys.stderr.getvalue()
            self.assertIn("WARNING", stderr_output)
            self.assertIn("points outside its directory", stderr_output)
            self.assertIn("../other/libcuda.so.565.57.01", stderr_output)
        finally:
            sys.stderr = old_stderr


if __name__ == "__main__":
    unittest.main()

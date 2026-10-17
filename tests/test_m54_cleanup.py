"""删除边界必须先于实际unlink核验；不操作服务器或科研文件。"""
import tempfile
import unittest
from pathlib import Path
from tools.m54_cleanup import safe_path


class CleanupBoundaryTests(unittest.TestCase):
    def test_protected_source_and_evidence_cannot_be_selected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); weight=root/'source.pt'; weight.write_bytes(b'fixture')
            with self.assertRaisesRegex(AssertionError,'PROTECTED_SOURCE'):
                safe_path(weight,root,{weight.resolve()})
            log=root/'failure.json'; log.write_text('{}')
            with self.assertRaisesRegex(AssertionError,'NOT_A_WEIGHT_OR_TRACE'):
                safe_path(log,root,set())

    def test_outside_root_cannot_be_deleted_and_valid_weight_can(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'experiment'; root.mkdir()
            outside=Path(folder)/'other.pt'; outside.write_bytes(b'fixture')
            with self.assertRaises(ValueError): safe_path(outside,root,set())
            valid=root/'retired.pt';valid.write_bytes(b'fixture')
            self.assertEqual(safe_path(valid,root,set()),valid.resolve())
            self.assertTrue(outside.exists());self.assertTrue(valid.exists())


if __name__=='__main__':unittest.main()

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from cn_ipa.common import write
from cn_ipa import scoring


class ProtocolTests(unittest.TestCase):
    def test_structure_boundary_precedes_lpips(self):
        base=dict(lpips_gt=.4,contour_f1=.7,iou=.8,leak=.02)
        safe=dict(lpips_gt=.35,contour_f1=.69,iou=.80,leak=.021)
        unsafe=dict(lpips_gt=.20,contour_f1=.60,iou=.80,leak=.021)
        self.assertLess(scoring.selection_key(safe,base),scoring.selection_key(unsafe,base))

    def test_identity_pairing_missing_seed_and_any_seed_extreme(self):
        with tempfile.TemporaryDirectory(prefix='cnipa-stat-',dir=Path.cwd()) as folder:
            root=Path(folder);self.assertEqual(root.resolve().parent,Path.cwd().resolve())
            write(root/'manifests/test_A.json',[dict(id=i) for i in ['a','b','c']])
            records=[]
            arms=['E5','B1','M10','M10_GAIN','M12','M12_GAIN','C0','C1','C2']
            for arm in arms:
                for identity in ['a','b','c']:
                    for seed in [42,43,44]:
                        r=dict(variant=arm,id=identity,seed=seed,lpips_gt=.4,contour_f1=.7,iou=.8,leak=.02)
                        if arm=='C1':r['lpips_gt']=.3
                        if arm=='C2' and identity=='a' and seed==43:r.update(lpips_gt=None,contour_f1=.49)
                        records.append(r)
            with patch.object(scoring,'ROOT',root):
                result=scoring.statistics(records)
            self.assertEqual(result['C2']['lpips_gt']['n'],2)
            paired=scoring.read(root/'metrics/paired_ci.json')
            delta=next(r for r in paired if r['candidate']=='C1' and r['baseline']=='C0' and r['metric']=='lpips_gt')
            self.assertEqual(delta['n'],3)
            self.assertAlmostEqual(delta['mean'],-.1)
            self.assertAlmostEqual(delta['ci95'][0],-.1)
            self.assertEqual(scoring.read(root/'metrics/failure_ids.json'),[dict(variant='C2',id='a',worst_delta=.49-.7)])


if __name__=='__main__':unittest.main()

"""检验伪重复、缺 Seed、NA 与单 Seed 极端事件的统计风险。"""
import unittest
from tools.m12_gain_confirm_protocol import ARMS,SEEDS,CORE,aggregate,statistics,non_clothing_caption


def fixture():
    return [dict(id=i,variant=a,seed=s,lpips_gt=.3-(.01 if a!='E5' else 0),contour_f1=.7,iou=.8,leak=.03)
        for a in ARMS for i in ['a','b','c'] for s in SEEDS]


class ConfirmStatistics(unittest.TestCase):
    def test_rolltop_and_drawstring_backpacks_are_not_clothing(self):
        self.assertTrue(non_clothing_caption('A denim backpack with rolled top, front zipper pocket.'))
        self.assertTrue(non_clothing_caption('A beige woven backpack with a drawstring top.'))
        self.assertFalse(non_clothing_caption('A white striped tank top with thin straps.'))

    def test_one_identity_three_seeds(self):
        records=fixture();records[0]['contour_f1']=.1
        rows=aggregate(records,['a','b','c'])
        value=next(r for r in rows if r['id']=='a' and r['variant']=='E5')
        self.assertAlmostEqual(value['contour_f1'],.5)
        self.assertEqual(len(rows),18)

    def test_missing_or_duplicate_seed_fails(self):
        records=fixture()
        with self.assertRaises(AssertionError):aggregate(records[:-1],['a','b','c'])
        records[-1]=dict(records[-2])
        with self.assertRaises(AssertionError):aggregate(records,['a','b','c'])

    def test_na_is_not_filled_or_other_seeds_averaged(self):
        records=fixture();records[0]['contour_f1']=None
        rows=aggregate(records,['a','b','c'])
        self.assertIsNone(next(r for r in rows if r['id']=='a' and r['variant']=='E5')['contour_f1'])

    def test_any_seed_extreme_cannot_hide_in_mean(self):
        records=fixture()
        for row in records:
            if row['variant']=='M12_GAIN' and row['id']=='a' and row['seed']==43:row['contour_f1']=.5
        result=statistics(records,['a','b','c'])
        self.assertEqual(result['paired']['E5']['contour_f1']['n'],3)
        self.assertEqual(len(result['gates']['M12_GAIN']['extreme_identity_events']),1)
        self.assertEqual(result['gates']['M12_GAIN']['mean_seed_extreme_ids'],[])
        self.assertFalse(result['gates']['M12_GAIN']['checks']['extreme'])


if __name__=='__main__':unittest.main()

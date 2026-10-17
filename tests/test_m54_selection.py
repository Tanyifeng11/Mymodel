"""验证预算边界和安全排序，不模拟或声称任何科研结果。"""
import unittest
from tools.m54_selection import select_pilot
from tools.m54_protocol import result_stage


def fixture(arm,f1=0.,iou=0.,lpips=-.04,tail=0):
    metrics={'contour_f1':dict(mean=f1,ci95=[f1-.01,f1+.01]),
             'iou':dict(mean=iou,ci95=[iou-.005,iou+.005]),'lpips_gt':dict(mean=lpips,ci95=[lpips-.01,lpips+.01])}
    gate=dict(paired_E5=metrics,extreme_events=[{}]*tail,regional_metrics_complete=True)
    return dict(lpips_gt=.3+lpips),gate,dict(variant=arm,parameters=100,latency_seconds=1,peak_vram_gib=1)


class SelectionTests(unittest.TestCase):
    def test_clutch_is_not_a_top_because_caption_mentions_top_clasp(self):
        from tools.m54_input_candidates import clothing
        self.assertIsNone(clothing(dict(caption='A white rectangular clutch with a top clasp.')))
        self.assertEqual(clothing(dict(caption='A striped sleeveless top with a round neckline.')),'clothing')

    def choose(self,definitions):
        means={};gates={};records=[]
        for arm,values in definitions.items():
            means[arm],gates[arm],record=fixture(arm,**values);records.append(record)
        return select_pilot(means,gates,records)

    def test_safety_beats_single_appearance_score_and_family_is_unique(self):
        r=self.choose({'M1':{},'M2':dict(tail=2,lpips=-.15),'M3':{},'M4':{},
                       'M5_S075':dict(f1=.01),'M5_S100':dict(f1=.02),'M5_S125':{},'M7_A0':{},'M8_T100':{}})
        self.assertTrue(r['ready']);self.assertEqual(len(r['selected']),4)
        self.assertEqual(r['selected'][0],'M5_S100');self.assertNotIn('M2',r['selected'])
        self.assertEqual(sum(a.startswith('M5') for a in r['selected']),1)
        self.assertNotIn('M7_A0',r['selected']);self.assertNotIn('M8_T100',r['selected'])

    def test_ambiguous_exception_cannot_be_silently_waived(self):
        r=self.choose({'M1':dict(lpips=.01,f1=.04),'M9':dict(f1=-.06,lpips=-.2),'M6':dict(tail=3)})
        self.assertFalse(r['ready']);self.assertEqual(set(r['ambiguous']),{'M1','M9'})
        self.assertIn('M6',r['rejected'])

    def test_deepening_and_seed_paths_are_distinct(self):
        self.assertEqual(result_stage('pilot'),'pilot')
        self.assertEqual(result_stage('natural',2400,123),'natural_u2400_s123')


if __name__=='__main__':unittest.main()

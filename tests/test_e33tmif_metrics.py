"""验证定位结论依赖的分母、配对和可读性归因，不测试模型实现细节。"""
import unittest
import numpy as np
from tools.e33tmif_metrics import pair,summarize,compare

class MetricsTest(unittest.TestCase):
    def test_rotation_and_missing_support(self):
        a=np.zeros((4,2,2));a[0]=1;a[3]=1
        b=a.copy();b[0]=-1
        support=np.ones((2,2),bool)
        out=pair(dict(R0=a,R90=b,R180=a),support,np.ones((2,2)))
        self.assertTrue(out['r90_success']);self.assertTrue(out['r180_success'])
        b[3]=0
        bad=pair(dict(R0=a,R90=b,R180=a),support,np.ones((2,2)))
        self.assertFalse(bad['r90_success']);self.assertIsNone(bad['r90_error'])
        rows=[dict(id=str(i),arms={k:{} for k in ('R0','R90','R180')},**r) for i,r in enumerate([out,bad])]
        stats=summarize(rows)
        self.assertEqual(stats['statistics']['r90_success']['mean'],.5)
        self.assertEqual(stats['statistics']['r90_success_readable']['mean'],1.)
        self.assertEqual(stats['r90_readable_N'],1)

    def test_readability_loss_is_distinct_from_wrong_orientation(self):
        def record(i,success,readable,error):
            return dict(id=str(i),r90_success=success,r180_success=True,r90_readable=readable,r180_readable=True,
                r90_coverage=float(readable),r180_coverage=1.,r90_error=error)
        before=[record(i,True,True,0.) for i in range(10)]
        after=[record(i,False,False,None) for i in range(6)]+[record(i,False,True,90.) for i in range(6,10)]
        out=compare(before,after)
        self.assertEqual(out['newly_unreadable_N'],6)
        self.assertEqual(out['still_readable_incorrect_N'],4)
        self.assertTrue(out['readability_loss']);self.assertTrue(out['significant_drop'])
        self.assertEqual(out['response_error_increase']['n'],4)
        self.assertEqual(out['response_error_increase']['mean'],90.)

    def test_duplicate_identity_is_rejected(self):
        with self.assertRaises(AssertionError): summarize([dict(id='a'),dict(id='a')])

if __name__=='__main__':unittest.main()

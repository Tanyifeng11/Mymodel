import unittest
import numpy as np
from tools.e14_pattern_attributes import folds_for, evaluate


class AttributeTests(unittest.TestCase):
    def setUp(self):
        self.params=np.array([(f,a,p) for f in (3,5,8,12) for a in (0,30) for p in (0,27)])

    def test_split_group_isolation(self):
        for task,column in [('frequency_unseen_scale',0),('orientation_unseen_scale',0),('frequency_cross_angle',1)]:
            tested=[]
            for _,train,test in folds_for(self.params,task):
                self.assertFalse(set(self.params[train,column]) & set(self.params[test,column]))
                tested.extend(test)
            self.assertEqual(sorted(tested),list(range(16)))

    def test_constant_baselines(self):
        x=np.ones((16,3))
        r=evaluate(x,self.params,'frequency_unseen_scale')
        self.assertAlmostEqual(r['mae_log2'],r['baseline_mae_log2'])
        self.assertEqual(evaluate(x,self.params,'orientation_unseen_scale')['balanced_accuracy'],.5)
        self.assertEqual(evaluate(x,self.params,'frequency_cross_angle')['balanced_accuracy'],.25)

    def test_readable_direction(self):
        x=self.params[:,1:2].astype(float)
        self.assertEqual(evaluate(x,self.params,'orientation_unseen_scale')['balanced_accuracy'],1)


if __name__=='__main__':
    unittest.main()

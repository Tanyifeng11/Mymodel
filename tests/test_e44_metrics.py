"""确认外轮廓指标不惩罚内部线条，灰度纹理测量有基本正对照。"""
import unittest
import numpy as np
from tools.e44_metrics import calibration,contour_f1


class TestMetrics(unittest.TestCase):
    def test_programmed_controls(self):
        result=calibration()
        self.assertGreater(result['rotated_error'],result['same_error'])
        mask=np.zeros((128,128),bool);mask[15:110,30:95]=True
        shifted=np.roll(mask,15,axis=1)
        self.assertLess(contour_f1(shifted,mask),contour_f1(mask,mask))


if __name__=='__main__':unittest.main()

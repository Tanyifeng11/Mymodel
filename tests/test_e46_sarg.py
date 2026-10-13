"""门控语义和几何的必要检查；真实层序/CFG/完整回放由 GPU P0 验收。"""
import unittest
import cv2
import numpy as np
import torch
from PIL import Image
from models.e46_sarg import boundary_band,gate_from_band,scale_residual


class TestSARG(unittest.TestCase):
    def test_external_contour_ignores_closed_hole(self):
        mask=np.zeros((512,384),np.uint8);mask[32:480,32:352]=255;mask[200:300,150:230]=0
        maps=boundary_band(Image.fromarray(mask))
        self.assertEqual(maps['info']['contour_count'],1)
        self.assertEqual(int(maps['edge'][200:300,150:230].sum()),0)
        self.assertEqual(float(maps['band'][32,100]),1.)
        self.assertLess(float(maps['band'][250,190]),1e-8)
        gate=gate_from_band(maps['small'])
        self.assertGreaterEqual(float(gate.min()),.26-1e-7)
        self.assertLessEqual(float(gate.max()),.65+1e-7)
        self.assertLess(float(gate[2,8]),float(gate[16,12]))
        self.assertTrue(np.array_equal(maps['small'],cv2.resize(maps['band'],(24,32),interpolation=cv2.INTER_AREA)))

    def test_projected_increment_only_zero_identity_constant(self):
        for dtype in [torch.float32,torch.float16]:
            base=torch.randn(1,768,4,dtype=dtype)+3
            delta=torch.randn_like(base)
            self.assertIs(scale_residual(base,delta,0),base)
            self.assertTrue(torch.equal(scale_residual(base,delta,1),base+delta))
            band=torch.rand(32,24)
            self.assertTrue(torch.equal(scale_residual(base,delta,gate_from_band(band,beta=0)),
                                       scale_residual(base,delta,.65)))
            gate=gate_from_band(band)
            output=scale_residual(base,delta,gate)
            self.assertFalse(torch.equal(output,(base+delta)*gate.to(dtype).reshape(1,768,1)))

    def test_corner_and_contour_coordinate_order(self):
        coordinate=torch.arange(768).reshape(1,1,32,24)
        tokens=coordinate.flatten(2).transpose(1,2)
        for y,x in [(0,0),(0,23),(31,0),(7,11)]:
            self.assertEqual(int(tokens[0,y*24+x,0]),y*24+x)
        base=torch.zeros(1,768,2);delta=torch.ones_like(base)
        gate=torch.arange(768).reshape(32,24)/768
        output=scale_residual(base,delta,gate)
        for y,x in [(0,0),(0,23),(31,0),(7,11)]:
            self.assertEqual(float(output[0,y*24+x,0]),float(gate[y,x]))


if __name__=='__main__':unittest.main()

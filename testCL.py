# -*- coding: utf-8 -*-
"""
Benchmark template matching with opencl
Use spyder profiler
to try on other devices too
"""

from skimage.data import coins
import cv2

#%% Get image and templates by cropping
image     = coins()
smallCoin = image[37:37+38, 80:80+41] 

imageCL, smallCoinCL  = map(cv2.UMat, [image, smallCoin])

def test():
   return cv2.matchTemplate(image, smallCoin, 0)

def testCL():
    return cv2.matchTemplate(imageCL, smallCoinCL, 0)
    
out   = test()
outCL = testCL()

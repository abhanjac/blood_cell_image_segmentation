# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 08:52:21 2019

This script contains all the common functions and global variables and training
parameters used by all the other scripts.

@author: abhanjac
"""

import cv2, numpy as np, os, time, datetime, copy, sys, json, random, shutil
import matplotlib.pyplot as plt

from utils_2 import *
from unet_model_6 import *

if __name__ == '__main__':

    trainDir = 'train'
    validDir = 'valid'
    testDir = 'test'
    trialDir = 'trial'
    inferDir = trialDir

    #unetSegmentor = unet1()
    
    calculateSegMapWeights( trainDir )
    
##-------------------------------------------------------------------------------

    #b, h, w, c = labelBatch.shape
    #print( labelBatch.shape )

    #for b1 in range(b):
        #img = imgBatch[b1]
        #cv2.imshow( 'img', img )
        
        #for cls in range(c):
            #chan = labelBatch[b1,:,:,cls]
            #if cls in classIdxToName:
                #clsName = classIdxToName[cls]
                #cv2.imshow( 'label channel {}'.format(clsName), chan )
            #else:
                #cv2.imshow( 'label channel {}'.format('Background'), chan )
            
        #cv2.waitKey(0)
        
    #cv2.destroyAllWindows()


#===============================================================================

    
    

    

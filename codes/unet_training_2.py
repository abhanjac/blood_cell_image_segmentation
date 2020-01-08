# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 08:52:32 2019

@author: abhanjac
"""

import tensorflow as tf

from utils_2 import *
from unet_model_6 import *

#===============================================================================

if __name__ == '__main__':
    
    trainDir = 'train'
    validDir = 'valid'
    testDir = 'test'
    trialDir = 'trial'
    
    unetSegmentor = unet1()
    unetSegmentor.train( trainDir=trainDir, validDir=validDir )
    #unetSegmentor.train( trainDir=trialDir, validDir=trialDir )


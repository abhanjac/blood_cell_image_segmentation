# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 14:28:52 2019

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

#-------------------------------------------------------------------------------
   
   unetSegmentor = unet1()
   unetSegmentor.saveLayer = False   # Activate the flag to save the layer outputs.
   inferDir = trialDir
   key = '`'
   listOfImg = os.listdir( os.path.join( inferDir, 'images' ) )
   listOfImg.sort()
   nImgs = len( listOfImg )

#-------------------------------------------------------------------------------

   for idx, i in enumerate( listOfImg ):
       
       ## Skip images if needed.
       #if idx < 23:   continue
       
       # Prediction from network.
       img = cv2.imread( os.path.join( inferDir, 'images', i ) )
       
       # Creating the label name from the image name.
       labelName = i.split('_')
       labelName.insert( -4, 'seg' )
       labelName = '_'.join( labelName )

       # Reading the segment image file.
       segLabel = cv2.imread( os.path.join( inferDir, 'segments', labelName ) )
       img1 = copy.deepcopy( img )
       imgBatch = np.array( [ img ] )

       inferLayerOut, inferPredLogits, inferPredProb, inferPredLabel, _, _ = \
                                           unetSegmentor.batchInference( imgBatch )
       # Removing the batch axis.
       inferPredLabel = inferPredLabel[0]
       
       # Converting the array from int32 to uint8.
       inferPredLabel = np.asarray( inferPredLabel, dtype=np.uint8 )
       h, w, nChan = inferPredLabel.shape
       
       # Creating a blank output image onto which the predicted channels will be
       # superimposed.
       predictedLabel = np.zeros( img.shape, dtype=np.uint8 )
       
#-------------------------------------------------------------------------------

       for c in range( nChan ):
           channel = inferPredLabel[:,:,c]
           channel = cv2.cvtColor( channel, cv2.COLOR_GRAY2BGR )
           
           # Color for the background is [0,0,0] which is the (nClasses+1)th channel.
           color = classIdxToColor[c] if c < nClasses else [0,0,0]
           channel = channel * np.array( color )    # Coloring the image.
           channel = np.array( channel, dtype=np.uint8 )    # Converting to np,uint8.
           predictedLabel += channel
           
           #cv2.imshow( 'channel', predictedLabel )
           #cv2.waitKey(0)
           
       predictedLabel = np.array( predictedLabel, dtype=np.uint8 )

#-------------------------------------------------------------------------------

       cv2.imshow( 'Image', img )
       cv2.imshow( 'Segment', segLabel )
       cv2.imshow( 'Predicted Output', predictedLabel )
       print( i )
       key = cv2.waitKey(1)
       if key & 0xFF == 27:    break    # Break with esc key.
       elif key & 0xFF == ord('s'):      # Save image.
           cv2.imwrite( 'predicted_'+ i, predictedLabel )
           #cv2.imwrite( i, np.hstack([img, segLabel]) )
           cv2.imwrite( labelName, segLabel )
       
   cv2.destroyAllWindows()


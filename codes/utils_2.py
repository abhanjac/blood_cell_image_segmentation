# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 08:52:21 2019

This script contains all the common functions and global variables and training
parameters used by all the other scripts.

@author: abhanjac
"""

import cv2, numpy as np, os, time, datetime, copy, sys, json, random, shutil
import matplotlib.pyplot as plt

#===============================================================================

# Global variables and parameters.

# Variables that will mark the points in the image by mouse click using the 
# callback function.
cBix, cBiy = -1, -1

#-------------------------------------------------------------------------------

# Variables for the classifier.

classNameToIdx = { 'Eosinophil': 0, 'Basophil': 1, 'Neutrophil': 2, \
                   'Lymphocytes': 3, 'Monocytes': 4, 'partialWBC': 5, \
                   'Thrombocytes': 6, 'Clumps': 7, 'Infected': 8, '_': 9 }

classIdxToName = { 0: 'Eosinophil', 1: 'Basophil', 2: 'Neutrophil', \
                   3: 'Lymphocytes', 4: 'Monocytes', 5: 'partialWBC', \
                   6: 'Thrombocytes', 7: 'Clumps', 8: 'Infected', 9: '_' }

nClasses = len( classIdxToName )
inImgH, inImgW = 224, 224
leak = 0.1      # Parameter for the leaky relu.
learningRate = 0.0001
batchSize = 50
nEpochs = 100
threshProb = 0.5    # Beyond this threshold, a label element is 1 else 0.
nSavedCkpt = 5      # Number of checkpoints to keep at a time.
modelSaveInterval = 1   # Number of epochs after which the model will be saved.
ckptDirPath = 'saved_models'    # Location where the model checkpoints are saved.
recordedMean = np.array( [ 233.89406, 223.98854, 233.20714 ] )  # Mean of train dataset.
recordedStd = np.array( [ 40.6225, 53.69572, 40.3735 ] )  # Std of train dataset.

modelName = 'tiny_yolo'
savedCkptName = modelName + '_ckpt'

#-------------------------------------------------------------------------------

# Colors meant for segmentation label images (if created).

classNameToColor = { 'Eosinophil':   [255,255,0],   'Basophil':   [128,128,0], \
                     'Neutrophil':   [190,190,250], 'Lymphocytes': [0,128,128], \
                     'Monocytes':    [128,0,0],     'partialWBC': [255,0,0],   \
                     'Thrombocytes': [0,225,255],   'Clumps':     [255,0,255], \
                     'Infected':     [0,0,255],     '_':        [0,255,0] }  # BGR format.

classIdxToColor = { 0: [255,255,0], 1: [128,128,0], \
                    2: [190,190,250],   3: [0,128,128], \
                    4: [128,0,0],   5: [255,0,0],   \
                    6: [0,225,255], 7: [255,0,255], \
                    8: [0,0,255],   9: [0,255,0] }  # BGR format.

colorDict = { 'red':   [0,0,255],    'green':   [0,255,0],     'yellow': [0,225,255],    \
              'blue':  [255,0,0],     'orange': [48,130,245], 'purple':   [180,30,145],  \
              'cyan':  [255,255,0],  'magenta': [255,0,255],   'lime':   [60,245,210],   \
              'pink':  [190,190,250], 'teal':   [128,128,0],  'lavender': [255,190,230], \
              'brown': [40,110,170], 'beige':   [200,250,255], 'maroon': [0,0,128],      \
              'mint':  [195,255,170], 'olive':  [0,128,128],  'coral':    [180,215,255], \
              'navy':  [128,0,0],    'grey':    [128,128,128], 'white':  [255,255,255],  \
              'black': [0,0,0] }  # BGR format.

# This dictionary gives the weights to be assigned to the pixels of different 
# colors, since the number of pixels of different colors is not the same which 
# may make the segmentation network biased toward any particular pixel color.
# The weight of each pixel is inversely proportional to the percentage of those 
# kind of pixels in the overall set of training images.
classIdxToSegColorWeight = { 0: 57.496, 1: 109.076, 2: 58.761, 3: 104.468, \
                             4: 42.830, 5: 163.219, 6: 1177.220, 7: 467.332, \
                             8: 7.553,  9: 7.948,   10: 1.524 }

# Average number of pixels per object. This can be used for counting the number
# of objects from a predicted segmentation map.
classIdxToAvgPixelsPerClassObj = { 0: 4986.603, 1: 4994.226, 2: 4879.284, 3: 2744.476, \
                                   4: 6694.133, 5: 1982.704, 6: 243.549, 7: 613.506, \
                                   8: 1092.446, 9: 1040.280 }

#-------------------------------------------------------------------------------

# Variables for the detector.

# List of anchor boxes (width, height). These sizes are relative to a 
# finalLayerH x finalLayerW sized image.
anchorList = [ [ 3.1, 3.1 ], [ 6.125, 3.1 ], [ 3.1, 6.125 ], \
               [ 6.125, 6.125 ], [ 5.32, 5.32 ], [ 2.7, 2.7 ] ]
nAnchors = len( anchorList )

ckptDirPathDetector = ckptDirPath + '_detector'
savedCkptNameDetector = modelName + '_detector' + '_ckpt'
finalLayerH, finalLayerW = 14, 14   # Dimension of the final conv layer activation map.

iouThresh = 0.7
iouThreshForMAPcalculation = 0.3
lambdaCoord = 5.0
lambdaNoObj = 0.5
lambdaClass = 2.5
threshProbDetection = 0.2    # Beyond this threshold, a label element is 1 else 0.

#-------------------------------------------------------------------------------

# Variables for the rbc cells.

noiseyContArea = 50
rbcDia = 50
rbcArea = np.pi * rbcDia * rbcDia / 4
rbcNucleusDia = 25
rbcNucleusArea = np.pi * rbcNucleusDia * rbcNucleusDia / 4

#===============================================================================

def horiFlipSampleAndMask( sample=None, mask=None ):
    '''
    Performs horizontal flips on input sample and mask.
    '''
    if sample is None or mask is None:
        print( '\nERROR: one or more input arguments missing ' \
               'in horiFlipSampleAndMask. Aborting.\n' )
        sys.exit()    

    newSample = cv2.flip( sample, 1 )      # Flip around y axis.
    newMask = cv2.flip( mask, 1 )

    return newSample, newMask

#===============================================================================

def vertFlipSampleAndMask( sample=None, mask=None ):
    '''
    Performs vertical flips on input sample and mask.
    '''
    if sample is None or mask is None:
        print( '\nERROR: one or more input arguments missing ' \
               'in vertFlipSampleAndMask. Aborting.\n' )
        sys.exit()    

    newSample = cv2.flip( sample, 0 )      # Flip around x axis.
    newMask = cv2.flip( mask, 0 )

    return newSample, newMask

#===============================================================================

def random90degFlipSampleAndMask( sample=None, mask=None ):
    '''
    Performs 90 deg flips on input sample and mask randomly.
    '''
    if sample is None or mask is None:
        print( '\nERROR: one or more input arguments missing ' \
               'in random90degFlipSampleAndMask. Aborting.\n' )
        sys.exit()    
        
    # Now the selection of whether the flip should be by 90, 180 or 270
    # deg, is done randomly (with equal probablity).
    number1 = np.random.randint( 100 )

    if number1 < 33:
        # Flip by 90 deg (same as horizontal flip + transpose).
        newSample = cv2.transpose( cv2.flip( sample, 1 ) )
        newMask = cv2.transpose( cv2.flip( mask, 1 ) )
        
    elif number1 >= 33 and number1 < 66:
        # Flip by 180 deg (same as horizontal flip + vertical flip).
        newSample = cv2.flip( sample, -1 )
        newMask = cv2.flip( mask, -1 )
        
    else:   # Flip by 270 deg (same as vertical flip + transpose).
        newSample = cv2.transpose( cv2.flip( sample, 0 ) )
        newMask = cv2.transpose( cv2.flip( mask, 0 ) )
        
    # Also, finding the bbox for the rotated sample from the mask.
    h, w, _ = newSample.shape

    return newSample, newMask, w, h

#===============================================================================

def randomRotationSampleAndMask( sample=None, mask=None ):
    '''
    Performs rotation of the sample by arbitrary angles.
    '''
    if sample is None or mask is None:
        print( '\nERROR: one or more input arguments missing' \
               'in randomRotationSampleAndMask. Aborting.\n' )
        sys.exit()    
        
    # During rotation by arbitrary angles, the sample first needs to be
    # pasted on a bigger blank array, otherwise it will get cropped 
    # due to rotation.
    sampleH, sampleW, _ = sample.shape
    
    # The length of the side of the new blank array should be equal to
    # the diagonal length of the sample, so that any rotation can be 
    # accomodated.
    sideLen = int( np.sqrt( sampleH **2 + sampleW **2 ) + 1 )
    blankArr = np.zeros( (sideLen, sideLen, 3), dtype=np.uint8 )
    
    # Top left corner x and y coordinates of the region where the 
    # sample will be affixed on the blank array.
    sampleTlY = int( ( sideLen - sampleH ) / 2 )
    sampleTlX = int( ( sideLen - sampleW ) / 2 )
    
    # Affixing the sample on the blank array.
    blankArr[ sampleTlY : sampleTlY + sampleH, \
              sampleTlX : sampleTlX + sampleW, : ] = sample
             
    newSample = copy.deepcopy( blankArr )
    
    # Rotation angle is determined at random between 0 to 360 deg.
    angle = np.random.randint( 360 )
    
    # Create the rotation matrix and rotate the sample.
    M = cv2.getRotationMatrix2D( ( sideLen/2, sideLen/2 ), angle, 1 )
    newSample = cv2.warpAffine( newSample, M, ( sideLen, sideLen ) )
    
    # Modifying the mask in the same manner.
    blankArr1 = np.zeros( (sideLen, sideLen, 3), dtype=np.uint8 )
    blankArr1[ sampleTlY : sampleTlY + sampleH, \
               sampleTlX : sampleTlX + sampleW, : ] = mask
    newMask = copy.deepcopy( blankArr1 )
    newMask = cv2.warpAffine( newMask, M, ( sideLen, sideLen ) )
    
    # Also, finding the bbox for the rotated sample from the mask contour.
    newMask1 = cv2.cvtColor( newMask, cv2.COLOR_BGR2GRAY )
    returnedTuple = cv2.findContours( newMask1, method=cv2.CHAIN_APPROX_SIMPLE, \
                                                mode=cv2.RETR_LIST )
    contours = returnedTuple[-2]
    
    x, y, w, h = cv2.boundingRect( contours[0] )

    return newSample, newMask, w, h

#===============================================================================

def fixSampleToBg( sample=None, mask=None, bg=None, tlX=None, tlY=None ):
    '''
    This function takes in a sample, its mask, a background and the top left 
    corner x and y coordinates of the region where the sample will be affixed 
    on the background. The background, sample and mask has to be in proper 
    shape and should have already undergone whatever data augmentation was 
    necessary. This function does not handle those processings.
    It also returns the center and bbox of the object after pasting.
    '''
    if sample is None or mask is None or bg is None or tlX is None or tlY is None:
        print( '\nERROR: one or more input arguments missing ' \
               'in fixSampleToBg. Aborting.\n' )
        sys.exit()    

    sampleH, sampleW, _ = sample.shape
    bgH, bgW, _ = bg.shape
    
    invMask = ( 255 - mask ) / 255
    
    # There are some bounding regions surrounding the actual object in the 
    # sample image. When this sample is affixed, we do not want these 
    # surrounding regions to replace the corresponding pixels in the bg image.
    # So the inverted mask is used to copy the pixels of the background 
    # corresponding to this bounding region and later paste those back after 
    # the sample has been affixed.
    
    # Now it may happen that the tlX and tlY are such that the sample will get
    # clipped at the image boundary. So determining the x and y coordinates of 
    # the bottom right corner as well.
    brY, brX = min( tlY + sampleH, bgH ), min( tlX + sampleW, bgW )
    
    bgRegionToBeReplaced = bg[ tlY : brY, tlX : brX, : ]
    bgRegTBRh, bgRegTBRw, _ = bgRegionToBeReplaced.shape
    
    # While taking out the bounding region (or the object only region), it is to 
    # be made sure that the size of these regions and the invMask (or mask) are 
    # same, otherwise this may throw errors in the cases where the sample is 
    # getting clipped.
    boundingRegion = bgRegionToBeReplaced * invMask[ 0 : bgRegTBRh, \
                                                     0 : bgRegTBRw, : ]
    boundingRegion = np.asarray( boundingRegion, dtype=np.uint8 )
    
    # Taking out only the object part of the sample using the mask.
    onlyObjectRegionOfSample = cv2.bitwise_and( sample, mask )
    onlyObjectRegionOfSample = onlyObjectRegionOfSample[ 0 : bgRegTBRh, \
                                                         0 : bgRegTBRw, : ]
    
    # Now pasting the sample onto the bg along with the pixels of bg 
    # corresponding to the blank region (which is called bounding region in this 
    # case).
    img = copy.deepcopy( bg )
    img[ tlY : brY, tlX : brX, : ] = onlyObjectRegionOfSample + boundingRegion
    
    # The location where the sample is affixed, this is the center pixel of this
    # region, not the top left corner.
    posY = round( (brY + tlY) * 0.5 )
    posX = round( (brX + tlX) * 0.5 )
    bboxH = brY - tlY
    bboxW = brX - tlX
    
    return img, posX, posY, bboxW, bboxH

#===============================================================================

def singleInstance( sampleLoc=None, maskLoc=None, \
                    bgLoc=None, imgSaveLoc=None, labelSaveLoc=None, \
                    nImgs=None, imgH=None, imgW=None,
                    saveNameSuffix=None, do90degFlips=False, \
                    doHoriFlip=False, doVertFlip=False, doRandomRot=False, \
                    clipSample=False, includeRbc=False, createSegmentLabelImg=False, \
                    segmentSaveLoc=None ):
    '''
    This function creates images where an object of a certain class appears 
    just one time.
    These images are created by taking object samples from sampleLoc and 
    backgrounds from bgLoc by affixing the object samples onto the backgrounds.
    The maskLoc holds the masks for the sample, but this is optional. If there
    is no maskLoc provided, then the samples are just pasted as they are, 
    otherwise the corresponding mask is used while pasting the sample.
    The samples and backgrounds are selected randomly from the available 
    collection in their respective locations. Total number of images created is 
    nImgs. These images are saved in the imgSaveLoc.
    The labels of the corresponding images are also created as json files in 
    the labelSaveLoc.
    imgH and imgW defines the size of the image to be created.
    The saveNameSuffix is a string, that will be appended to the name of the 
    image file while saving. This is important to identify the dataset from 
    where the image has been synthesized.
    The flag do90degFlips indicates whether the sample should undergo rotations
    by multiples of 90 deg (randomly), while getting affixed on the bg image.
    The flag doRandomRot indicates whether the sample should undergo rotations
    by random angles, while getting affixed on the bg image.
    Flags doHoriFlip and doVertFlip indicates if the sample should be flipped 
    horizontally or vertically (randomly) before getting affixed on bg image.
    The clipSample flag indicates if there will be any clipping of the sample
    when it is affixed on the bg.
    The function also checks if there are already some rbc annotations present 
    on the background or not. If so then it includes them in the labels.
    The createSegmentLabelImg indicates if a semantic segmentation label image 
    has to be created as well. The colors of the segments for different objects 
    are mentioned in the global variables. Segment save location is also provided.
    '''
    
    if sampleLoc is None or bgLoc is None \
       or imgSaveLoc is None or labelSaveLoc is None or nImgs is None \
       or imgH is None or imgW is None or saveNameSuffix is None:
           print( '\nERROR: one or more input arguments missing ' \
                  'in singleInstance. Aborting.\n' )
           sys.exit()
           
    if createSegmentLabelImg:
        if segmentSaveLoc is None:
            print( '\nERROR: one or more input arguments missing ' \
                   'in singleInstance for segments. Aborting.\n' )
            sys.exit()
    
    # Flag indicating mask present.
    maskPresent = False if maskLoc is None else True
    
#-------------------------------------------------------------------------------
        
    # Checking if there is any label file for the bg images present or not. 
    # These files will include rbc annotations, if present.
    imgFolderParentDir = '\\'.join( bgLoc.split('\\')[:-1] )
    imgFolderName = bgLoc.split('\\')[-1]
    labelFolderName = imgFolderName + '_labels'
    labelFolderLoc = os.path.join( imgFolderParentDir, labelFolderName )
    
    if createSegmentLabelImg:
        bgSegmentFolderName = imgFolderName + '_segments'
        bgSegmentFolderLoc = os.path.join( imgFolderParentDir, bgSegmentFolderName )

#-------------------------------------------------------------------------------
    
    # Number of files already existing in the imgSaveLoc is calculated. This 
    # will be used to assign the index to the file while saving.
    nAlreadyExistingFiles = len( os.listdir( imgSaveLoc ) )
    
    bgList, sampleList = [], []
    
    # Creating the images.    
    for i in range( nImgs ):
        # Fill the lists if they are empty.
        # As a sample and a bg is used for creating an image, they are deleted
        # from this list. So if this list gets empty, then it is reinitialized.
        if len( bgList ) == 0:      bgList = os.listdir( bgLoc )
        if len( sampleList ) == 0:      sampleList = os.listdir( sampleLoc )
        
        # Select a sample at random.
        sampleIdx = np.random.randint( len( sampleList ) )
        
        sampleName = sampleList[ sampleIdx ]
        sample = cv2.imread( os.path.join( sampleLoc, sampleName ) )
        
        className = sampleName.split('_')[0]
        
        if maskPresent:     
            # If name of sample is Eosinophil_1.bmp, the name of the 
            # corresponding mask is Eosinophil_1_mask.bmp
            maskName = sampleName[:-4] + '_mask.bmp'
            mask = cv2.imread( os.path.join( maskLoc, maskName ) )
        else:
            # If mask is not present then a dummy mask is created which is just
            # a blank array of 255s, of the same type and shape as sample.
            # This makes all future processing easier and also prevents the 
            # check for maskPresent flag every time.
            mask = np.ones( sample.shape ) * 255
            mask = np.asarray( mask, dtype=np.uint8 )
        
        # The bg and sample idxs are determined separately because the number of
        # available samples and bg may be different.
        bgIdx = np.random.randint( len( bgList ) )
        bgName = bgList[ bgIdx ]
        bg = cv2.imread( os.path.join( bgLoc, bgName ) )
        
        # Remove the entry of this sample and bg from the respective lists so 
        # that they are not used again. It will only be used again if all the 
        # existing samples in the lists are used up and the lists become empty.
        sampleList.pop( sampleIdx )
        bgList.pop( bgIdx )

#-------------------------------------------------------------------------------
        
        # Setting the background.
        
        # It may happen that the bg image is larger than size imgH x imgW.
        # In that case, a imgH x imgW region is cropped out from the bg image.
        bgH, bgW, _ = bg.shape
        
        # Determining the x and y of the top left corner of the region to be
        # cropped out from the bg image.
        bgTlY = np.random.randint( bgH - imgH ) if bgH > imgH else 0
        bgTlX = np.random.randint( bgW - imgW ) if bgW > imgW else 0
        
        # IMPORTANT: The bg image must be larger or equal in size to imgH x imgW.
        newBg = bg[ bgTlY : bgTlY + imgH, bgTlX : bgTlX + imgW ]
        
        newBgH, newBgW, _ = newBg.shape
        
#-------------------------------------------------------------------------------

        # Also doing the same processing for the segmented image label.
        if createSegmentLabelImg:
            bgSegName = 'seg_' + '_'.join( bgName.split('_')[1:] )
            bgSegImg = cv2.imread( os.path.join( bgSegmentFolderLoc, bgSegName ) )
            
            newBgSegImg = bgSegImg[ bgTlY : bgTlY + imgH, bgTlX : bgTlX + imgW ]
            
#-------------------------------------------------------------------------------

        # Including the rbc annotations in the label if there are label files 
        # present for these background images.

        # If however the argument includeRbc is false, this function will keep 
        # this dictionary empty.
        if includeRbc:
            # Now those rbc which falls within the region which is cropped for 
            # bg image, are included inside another dictionary.
            # If there are no rbc annotations, then this dictionary will stay empty.
            rbcOnCurrentBg = {}
    
            if os.path.exists( labelFolderLoc ):
                bgLabelName = bgName[:-4] + '.json'
                bgLabelLoc = os.path.join( labelFolderLoc, bgLabelName )
                
                with open( bgLabelLoc, 'r' ) as infoFile:
                    infoDict = json.load( infoFile )
                
                for k, v in infoDict.items():
                    posX, posY = v['posX'], v['posY']
                    if posX >= bgTlX and posX < bgTlX + imgW and \
                       posY >= bgTlY and posY < bgTlY + imgH:
                           rbcOnCurrentBg[k] = v
                           rbcOnCurrentBg[k]['posX'] -= bgTlX
                           rbcOnCurrentBg[k]['posY'] -= bgTlY
                           rbcOnCurrentBg[k]['tlX'] -= bgTlX
                           rbcOnCurrentBg[k]['tlY'] -= bgTlY
                           
        else:
            rbcOnCurrentBg = {}
                       
#-------------------------------------------------------------------------------

        # Clip sample at the image boundary.

        # If the clipSample flag is True, then the other data augmentation like
        # flipping and rotation is ignored (even if their corresponding flags
        # are True). As this does not make much of a difference.

        if clipSample:
            # All clipped samples will have the common name 'partialWBC'.
            className = 'partialWBC'
            
            newSample, newMask = sample, mask

            # Whether the clipping will happen at the side or the corner of the
            # image, will be again selected randomly.
            number2 = np.random.randint( 100 )
                
            # The sample will be affixed in a location on the bg, such that 
            # it gets clipped by half. The clipping is always done by half 
            # because, what matters during this clipping is that, the sample 
            # should only be visible by a variable amount inside the image.
            # Now because of the variation of the size of the samples, they 
            # will anyway be visible by variable amount inside the image 
            # even if the percentage of clipping is kept constant. So to 
            # keep things simple the clipping is always done by 50%.
            # Because of the same reason as stated in the previous case, 
            # the clipping at the corners is kept constant at 25% only. The 
            # variability in size of the samples will take care of the rest.
            
            newSampleH, newSampleW, _ = newSample.shape

            tlX = newBgW - int( newSampleW * 0.5 )
                
            if number2 < 15:
                # Clip at the left side of the image.
                tlY = np.random.randint( newBgH - newSampleH )
                # Fixing the sample onto the background.
                image, posX, posY, bboxW, bboxH = fixSampleToBg( newSample, newMask, newBg, tlX, tlY )
                # Now remove the rbc cells which are overlapped by this wbc cell.
                newRbcOnCurrentBg = { k : v for k, v in rbcOnCurrentBg.items() \
                                      if abs(posX-v['posX']) > bboxW/2 or abs(posY-v['posY']) > bboxH/2 }
                rbcOnCurrentBg = newRbcOnCurrentBg
                
                # Create the segmented label image as well if createSegmentLabelImg is True:
                if createSegmentLabelImg:
                    sampleColor = classNameToColor[ className ]
                    sampleSegImg = cv2.bitwise_and( np.array( sampleColor ), newMask )
                    segImg, _, _, _, _ = fixSampleToBg( sampleSegImg, newMask, newBgSegImg, tlX, tlY )
                    

            elif number2 >= 15 and number2 < 30:
                # Clip at the top side of the image (which is same as clip on 
                # left + flip by 90 deg).
                tlY = np.random.randint( newBgH - newSampleH )
                # Fixing the sample onto the background.
                image, posX, posY, bboxW, bboxH = fixSampleToBg( newSample, newMask, newBg, tlX, tlY )
                # Now remove the rbc cells which are overlapped by this wbc cell.
                newRbcOnCurrentBg = { k : v for k, v in rbcOnCurrentBg.items() \
                                      if abs(posX-v['posX']) > bboxW/2 or abs(posY-v['posY']) > bboxH/2 }
                rbcOnCurrentBg = newRbcOnCurrentBg

                image = cv2.transpose( cv2.flip( image, 1 ) )
                posX, posY, = posY, imgW - posX
                bboxW, bboxH = bboxH, bboxW
                # Now modifying the rbc cell locations.
                for k, v in rbcOnCurrentBg.items():
                    v['posX'], v['posY'] = v['posY'], imgW - v['posX']
                    v['bboxW'], v['bboxH'] = v['bboxH'], v['bboxW']

                # Create the segmented label image as well if createSegmentLabelImg is True:
                if createSegmentLabelImg:
                    sampleColor = classNameToColor[ className ]
                    sampleSegImg = cv2.bitwise_and( np.array( sampleColor ), newMask )
                    segImg, _, _, _, _ = fixSampleToBg( sampleSegImg, newMask, newBgSegImg, tlX, tlY )
                    segImg = cv2.transpose( cv2.flip( segImg, 1 ) )
                    

            elif number2 >= 30 and number2 < 45:
                # Clip at the right side of the image (which is same as clip on 
                # left + flip by 180 deg).
                tlY = np.random.randint( newBgH - newSampleH )
                # Fixing the sample onto the background.
                image, posX, posY, bboxW, bboxH = fixSampleToBg( newSample, newMask, newBg, tlX, tlY )
                # Now remove the rbc cells which are overlapped by this wbc cell.
                newRbcOnCurrentBg = { k : v for k, v in rbcOnCurrentBg.items() \
                                      if abs(posX-v['posX']) > bboxW/2 or abs(posY-v['posY']) > bboxH/2 }
                rbcOnCurrentBg = newRbcOnCurrentBg

                image = cv2.flip( image, -1 )
                posX, posY, = imgW - posX, imgH - posY
                # Now modifying the rbc cell locations.
                for k, v in rbcOnCurrentBg.items():
                    v['posX'], v['posY'] = imgW - v['posX'], imgH- v['posY']
            
                # Create the segmented label image as well if createSegmentLabelImg is True:
                if createSegmentLabelImg:
                    sampleColor = classNameToColor[ className ]
                    sampleSegImg = cv2.bitwise_and( np.array( sampleColor ), newMask )
                    segImg, _, _, _, _ = fixSampleToBg( sampleSegImg, newMask, newBgSegImg, tlX, tlY )
                    segImg = cv2.flip( segImg, -1 )
                    

            elif number2 >= 45 and number2 < 60:
                # Clip at the bottom side of the image (which is same as clip on 
                # left + flip by 270 deg).
                tlY = np.random.randint( newBgH - newSampleH )
                # Fixing the sample onto the background.
                image, posX, posY, bboxW, bboxH = fixSampleToBg( newSample, newMask, newBg, tlX, tlY )
                # Now remove the rbc cells which are overlapped by this wbc cell.
                newRbcOnCurrentBg = { k : v for k, v in rbcOnCurrentBg.items() \
                                      if abs(posX-v['posX']) > bboxW/2 or abs(posY-v['posY']) > bboxH/2 }
                rbcOnCurrentBg = newRbcOnCurrentBg

                image = cv2.transpose( cv2.flip( image, 0 ) )
                posX, posY, = imgH - posY, posX
                bboxW, bboxH = bboxH, bboxW
                # Now modifying the rbc cell locations.
                for k, v in rbcOnCurrentBg.items():
                    v['posX'], v['posY'] = imgH - v['posY'], v['posX']
                    v['bboxW'], v['bboxH'] = v['bboxH'], v['bboxW']

                # Create the segmented label image as well if createSegmentLabelImg is True:
                if createSegmentLabelImg:
                    sampleColor = classNameToColor[ className ]
                    sampleSegImg = cv2.bitwise_and( np.array( sampleColor ), newMask )
                    segImg, _, _, _, _ = fixSampleToBg( sampleSegImg, newMask, newBgSegImg, tlX, tlY )
                    segImg = cv2.transpose( cv2.flip( segImg, 0 ) )

#-------------------------------------------------------------------------------

            elif number2 >= 60 and number2 < 70:
                # Clip at the bottom right corner.
                tlY = newBgH - int( newSampleH * 0.5 )
                # Fixing the sample onto the background.
                image, posX, posY, bboxW, bboxH = fixSampleToBg( newSample, newMask, newBg, tlX, tlY )
                # Now remove the rbc cells which are overlapped by this wbc cell.
                newRbcOnCurrentBg = { k : v for k, v in rbcOnCurrentBg.items() \
                                      if abs(posX-v['posX']) > bboxW/2 or abs(posY-v['posY']) > bboxH/2 }
                rbcOnCurrentBg = newRbcOnCurrentBg
                
                # Create the segmented label image as well if createSegmentLabelImg is True:
                if createSegmentLabelImg:
                    sampleColor = classNameToColor[ className ]
                    sampleSegImg = cv2.bitwise_and( np.array( sampleColor ), newMask )
                    segImg, _, _, _, _ = fixSampleToBg( sampleSegImg, newMask, newBgSegImg, tlX, tlY )
                    

            elif number2 >= 70 and number2 < 80:
                # Clip at the top right corner (which is same as clip on the 
                # bottom right corner + flip by 90 deg). 
                tlY = newBgH - int( newSampleH * 0.5 )
                # Fixing the sample onto the background.
                image, posX, posY, bboxW, bboxH = fixSampleToBg( newSample, newMask, newBg, tlX, tlY )
                # Now remove the rbc cells which are overlapped by this wbc cell.
                newRbcOnCurrentBg = { k : v for k, v in rbcOnCurrentBg.items() \
                                      if abs(posX-v['posX']) > bboxW/2 or abs(posY-v['posY']) > bboxH/2 }
                rbcOnCurrentBg = newRbcOnCurrentBg

                image = cv2.transpose( cv2.flip( image, 1 ) )
                posX, posY, = posY, imgW - posX
                bboxW, bboxH = bboxH, bboxW
                # Now modifying the rbc cell locations.
                for k, v in rbcOnCurrentBg.items():
                    v['posX'], v['posY'] = v['posY'], imgW - v['posX']
                    v['bboxW'], v['bboxH'] = v['bboxH'], v['bboxW']
                    
                # Create the segmented label image as well if createSegmentLabelImg is True:
                if createSegmentLabelImg:
                    sampleColor = classNameToColor[ className ]
                    sampleSegImg = cv2.bitwise_and( np.array( sampleColor ), newMask )
                    segImg, _, _, _, _ = fixSampleToBg( sampleSegImg, newMask, newBgSegImg, tlX, tlY )
                    segImg = cv2.transpose( cv2.flip( segImg, 1 ) )


            elif number2 >= 80 and number2 < 90:
                # Clip at the top left corner (which is same as clip on the 
                # bottom right corner + flip by 180 deg). 
                tlY = newBgH - int( newSampleH * 0.5 )
                # Fixing the sample onto the background.
                image, posX, posY, bboxW, bboxH = fixSampleToBg( newSample, newMask, newBg, tlX, tlY )
                # Now remove the rbc cells which are overlapped by this wbc cell.
                newRbcOnCurrentBg = { k : v for k, v in rbcOnCurrentBg.items() \
                                      if abs(posX-v['posX']) > bboxW/2 or abs(posY-v['posY']) > bboxH/2 }
                rbcOnCurrentBg = newRbcOnCurrentBg

                image = cv2.flip( image, -1 )
                posX, posY, = imgW - posX, imgH - posY
                # Now modifying the rbc cell locations.
                for k, v in rbcOnCurrentBg.items():
                    v['posX'], v['posY'] = imgW - v['posX'], imgH - v['posY']

                # Create the segmented label image as well if createSegmentLabelImg is True:
                if createSegmentLabelImg:
                    sampleColor = classNameToColor[ className ]
                    sampleSegImg = cv2.bitwise_and( np.array( sampleColor ), newMask )
                    segImg, _, _, _, _ = fixSampleToBg( sampleSegImg, newMask, newBgSegImg, tlX, tlY )
                    segImg = cv2.flip( segImg, -1 )


            elif number2 >= 90 and number2 < 100:
                # Clip at the bottom left corner (which is same as clip on the 
                # bottom right corner + flip by 270 deg). 
                tlY = newBgH - int( newSampleH * 0.5 )
                # Fixing the sample onto the background.
                image, posX, posY, bboxW, bboxH = fixSampleToBg( newSample, newMask, newBg, tlX, tlY )
                # Now remove the rbc cells which are overlapped by this wbc cell.
                newRbcOnCurrentBg = { k : v for k, v in rbcOnCurrentBg.items() \
                                      if abs(posX-v['posX']) > bboxW/2 or abs(posY-v['posY']) > bboxH/2 }
                rbcOnCurrentBg = newRbcOnCurrentBg

                image = cv2.transpose( cv2.flip( image, 0 ) )
                posX, posY, = imgH - posY, posX
                bboxW, bboxH = bboxH, bboxW
                # Now modifying the rbc cell locations.
                for k, v in rbcOnCurrentBg.items():
                    v['posX'], v['posY'] = imgH - v['posY'], v['posX']
                    v['bboxW'], v['bboxH'] = v['bboxH'], v['bboxW']

                # Create the segmented label image as well if createSegmentLabelImg is True:
                if createSegmentLabelImg:
                    sampleColor = classNameToColor[ className ]
                    sampleSegImg = cv2.bitwise_and( np.array( sampleColor ), newMask )
                    segImg, _, _, _, _ = fixSampleToBg( sampleSegImg, newMask, newBgSegImg, tlX, tlY )
                    segImg = cv2.transpose( cv2.flip( segImg, 0 ) )

#-------------------------------------------------------------------------------
                
        # If the clipSample is False, then the other augmentations like flipping
        # and rotations are done.
        
        elif doHoriFlip or doVertFlip or do90degFlips or doRandomRot:

            # Augmenting the samples before affixing onto the background.
            
            # There are altogether 4 kinds of augmentation that this function 
            # can do, doRandomRot, doHoriFlip, doVertFlip, do90degFlips and no 
            # augmentation.
            # What kind of augmentation is to be done for this sample is chosen 
            # at random with a equal probability (20% for each type).
            # However, if the type of augmentation chosen doen not have it's 
            # corresponding flag True, then no augmentation is done.
            
            number = np.random.randint( 100 )
            
            # Horizontal flip.
            
            if number < 20 and doHoriFlip:
                newSample, newMask = horiFlipSampleAndMask( sample, mask )
                bboxH, bboxW, _ = newSample.shape

#-------------------------------------------------------------------------------

            # Vertical flip.
    
            elif number >= 20 and number < 40 and doVertFlip:
                newSample, newMask = vertFlipSampleAndMask( sample, mask )
                bboxH, bboxW, _ = newSample.shape
                
#-------------------------------------------------------------------------------
    
            # 90 deg flip.
    
            elif number >= 40 and number < 60 and do90degFlips:
                # Now the selection of whether the flip should be by 90, 180 or 270
                # deg, is done randomly (with equal probablity).
                newSample, newMask, bboxW, bboxH = random90degFlipSampleAndMask( sample, mask )
                
#-------------------------------------------------------------------------------

            # Rotation by random angles.
    
            elif number >= 60 and number < 80 and doRandomRot:
                # During rotation by arbitrary angles, the sample first needs to be
                # pasted on a bigger blank array, otherwise it will get cropped 
                # due to rotation.
                newSample, newMask, bboxW, bboxH = randomRotationSampleAndMask( sample, mask )
                
#-------------------------------------------------------------------------------
            
            # No augmentation.
    
            else:
                newSample, newMask = sample, mask
                bboxH, bboxW, _ = newSample.shape
            
            # x, y of top left corner of the region where sample will be pasted.
            newSampleH, newSampleW, _ = newSample.shape
            tlY = np.random.randint( newBgH - newSampleH )
            tlX = np.random.randint( newBgW - newSampleW )
                
            # Fixing the sample onto the background.
            image, posX, posY, _, _ = fixSampleToBg( newSample, newMask, newBg, tlX, tlY )
            # Now remove the rbc cells which are overlapped by this wbc cell.
            newRbcOnCurrentBg = { k : v for k, v in rbcOnCurrentBg.items() \
                                  if abs(posX-v['posX']) > bboxW/2 or abs(posY-v['posY']) > bboxH/2 }
            rbcOnCurrentBg = newRbcOnCurrentBg
            
            # Create the segmented label image as well if createSegmentLabelImg is True:
            if createSegmentLabelImg:
                sampleColor = classNameToColor[ className ]
                sampleSegImg = cv2.bitwise_and( np.array( sampleColor ), newMask )
                segImg, _, _, _, _ = fixSampleToBg( sampleSegImg, newMask, newBgSegImg, tlX, tlY )
                
#-------------------------------------------------------------------------------
        
        # If both the clipSample and the other augmentation flags are False, 
        # then no augmentation is performed.

        else:
            newSample, newMask = sample, mask
        
            # x, y of top left corner of the region where sample will be pasted.
            newSampleH, newSampleW, _ = newSample.shape
            tlY = np.random.randint( newBgH - newSampleH )
            tlX = np.random.randint( newBgW - newSampleW )
            # Fixing the sample onto the background.
            image, posX, posY, bboxW, bboxH = fixSampleToBg( newSample, newMask, newBg, tlX, tlY )
            # Now remove the rbc cells which are overlapped by this wbc cell.
            newRbcOnCurrentBg = { k : v for k, v in rbcOnCurrentBg.items() \
                                  if abs(posX-v['posX']) > bboxW/2 or abs(posY-v['posY']) > bboxH/2 }
            rbcOnCurrentBg = newRbcOnCurrentBg
            
            # Create the segmented label image as well if createSegmentLabelImg is True:
            if createSegmentLabelImg:
                sampleColor = classNameToColor[ className ]
                sampleSegImg = cv2.bitwise_and( np.array( sampleColor ), newMask )
                segImg, _, _, _, _ = fixSampleToBg( sampleSegImg, newMask, newBgSegImg, tlX, tlY )

##-------------------------------------------------------------------------------
#
#        cv2.imshow( 'sample', sample )
#        cv2.imshow( 'newSample', newSample )
#        cv2.imshow( 'mask', mask )
#        cv2.imshow( 'newMask', newMask )
#        cv2.imshow( 'newBg', newBg )
#        cv2.waitKey(0)
#                     
#-------------------------------------------------------------------------------

        # Saving the image.
        idx = nAlreadyExistingFiles + i    # This is the image index.
        
        imgSaveName = className[:4] + '_' + \
                      saveNameSuffix + '_' + str( idx ) + '.bmp'
        cv2.imwrite( os.path.join( imgSaveLoc, imgSaveName ), image )
        
        # Saving the segmented image label as well if createSegmentLabelImg is True.
        if createSegmentLabelImg:
            segImgSaveName = className[:4] + '_seg' + '_' + \
                             saveNameSuffix + '_' + str( idx ) + '.bmp'
            cv2.imwrite( os.path.join( segmentSaveLoc, segImgSaveName ), segImg )
        
        # Creating the label json file.
        labelSaveName = className[:4] + '_' + \
                        saveNameSuffix + '_' + str( idx ) + '.json'
        
        classIdx = classNameToIdx[ className ]
        
        infoDict = {}
        
        with open( os.path.join( labelSaveLoc, labelSaveName ), 'w' ) as infoFile:
            
#-------------------------------------------------------------------------------

            # Make sure the coordinates are inside the boundaries of the image.
            if posX >= imgW:      posX = imgW - 1
            if posX < 0:            posX = 0
            if posY >= imgH:      posY = imgH - 1
            if posY < 0:            posY = 0
            tlX, tlY = posX-bboxW*0.5, posY-bboxH*0.5   # Top left corner.
            brX, brY = posX+bboxW*0.5, posY+bboxH*0.5   # Bottom right corner.
            if tlX < 0:            tlX = 0
            if tlY < 0:            tlY = 0
            if brX >= imgW:      brX = imgW - 1
            if brY >= imgH:      brY = imgH - 1
            bboxW, bboxH = int( brX - tlX ), int( brY - tlY )   # Update box size.
            
#-------------------------------------------------------------------------------

            infoDict[0] = {
                            'className': className, 'classIdx': classIdx, \
                            'posX': int(posX), 'posY': int(posY), \
                            'bboxW': bboxW, 'bboxH': bboxH, \
                            'tlX': int(tlX), 'tlY': int(tlY), \
                            'samplePath': os.path.join( sampleLoc, sampleName ), \
                            'bgPath': os.path.join( bgLoc, bgName ) \
                          }
            
#-------------------------------------------------------------------------------

            # Now recording the rbc cells into the infoDict.
            nWbc = len( infoDict )      # Number of wbc cell records in infoDict.
#            print( len(rbcOnCurrentBg) )
            
            for r, (k, v) in enumerate( rbcOnCurrentBg.items() ):
                # Creating key for the rbc cell records. 
                # This makes sure that they are different from the keys of the 
                # wbc cell records, else they may overlap the wbc record.
                index = r + nWbc
                
#-------------------------------------------------------------------------------

                posX, posY, bboxW, bboxH = v['posX'], v['posY'], v['bboxW'], v['bboxH']
                classNameRbc, classIdxRbc = v['className'], v['classIdx']
                
#-------------------------------------------------------------------------------

                # Make sure the coordinates are inside the boundaries of the image.
                if posX >= imgW:      posX = imgW - 1
                if posX < 0:            posX = 0
                if posY >= imgH:      posY = imgH - 1
                if posY < 0:            posY = 0
                tlX, tlY = posX-bboxW*0.5, posY-bboxH*0.5   # Top left corner.
                brX, brY = posX+bboxW*0.5, posY+bboxH*0.5   # Bottom right corner.
                if tlX < 0:            tlX = 0
                if tlY < 0:            tlY = 0
                if brX >= imgW:      brX = imgW - 1
                if brY >= imgH:      brY = imgH - 1
                bboxW, bboxH = int( brX - tlX ), int( brY - tlY )   # Update box size.
            
#-------------------------------------------------------------------------------

                infoDict[ index ] = {
                                        'className': classNameRbc, 'classIdx': classIdxRbc, \
                                        'posX': int(posX), 'posY': int(posY), \
                                        'bboxW': bboxW, 'bboxH': bboxH, \
                                        'tlX': int(tlX), 'tlY': int(tlY), \
                                        'samplePath': None, \
                                        'bgPath': os.path.join( bgLoc, bgName ) \
                                    }
                
            json.dump( infoDict, infoFile, indent=4, separators=(',', ': ') )

#-------------------------------------------------------------------------------

        for k, v in infoDict.items():
            cv2.circle( image, (v['posX'], v['posY']), 2, (0,255,0), 2 )
            if v['className'] != '_':
                cv2.rectangle( image, (v['tlX'], v['tlY']), (v['tlX']+v['bboxW'], \
                                       v['tlY']+v['bboxH']), (0,255,0), 2 )
        cv2.imshow( 'image', image )
        # Show the segment label as well if the createSegmentLabelImg is True.
        if createSegmentLabelImg:   cv2.imshow( 'segment label', segImg )
        cv2.waitKey(30)

#===============================================================================

def tripleInstance( sampleLoc1=None, sampleLoc2=None, sampleLoc3=None, \
                    maskLoc1=None, maskLoc2=None, maskLoc3=None, bgLoc=None, \
                    imgSaveLoc=None, labelSaveLoc=None, nImgs=None, imgH=None, \
                    imgW=None, saveNameSuffix=None, clipSample1=False, \
                    clipSample2=False, clipSample3=False, includeRbc=False ):
    '''
    This function creates images where an object from each of the folders 
    sampleLoc1, sampleLoc2 and sampleLoc3 are randomly selected and pasted on a 
    background seleced from the bgLoc folder. So there are two instances of wbc 
    in the same image.
    The maskLoc1, maskLoc2 and maskLoc3 holds the masks for the sample of 
    sampleLoc1, sampleLoc2 and sampleLoc3 respectively, but these are optional. 
    If there are no maskLoc1 or maskLoc2 or maskLoc3 provided, then the 
    corresponding samples are just pasted as they are, otherwise the corresponding 
    mask is used while pasting the sample.
    The samples and backgrounds are selected randomly from the available 
    collection in their respective locations. Total number of images created is 
    nImgs. These images are saved in the imgSaveLoc.
    The labels of the corresponding images are also created as json files in 
    the labelSaveLoc.
    imgH and imgW defines the size of the image to be created.
    The saveNameSuffix is a string, that will be appended to the name of the 
    image file while saving. This is important to identify the dataset from 
    where the image has been synthesized.
    The clipSample1 (or clipSample2 or clipSample3) flag indicates if there will 
    be any clipping of the sample1 (or sample2 or sample3) when it is affixed on 
    the bg.
    The function also checks if there are already some rbc annotations present 
    on the background or not. If so then it includes them in the labels.
    '''
    
    if sampleLoc1 is None or sampleLoc2 is None or sampleLoc3 is None \
       or bgLoc is None or imgSaveLoc is None or labelSaveLoc is None \
       or nImgs is None or imgH is None or imgW is None or saveNameSuffix is None:
           print( '\nERROR: one or more input arguments missing ' \
                  'in tripleInstance. Aborting.\n' )
           sys.exit()
    
    # Flag indicating mask present.
    maskPresent1 = False if maskLoc1 is None else True
    maskPresent2 = False if maskLoc2 is None else True
    maskPresent3 = False if maskLoc3 is None else True
    
#-------------------------------------------------------------------------------
        
    # Checking if there is any label file for the bg images present or not. 
    # These files will include rbc annotations, if present.
    imgFolderParentDir = '\\'.join( bgLoc.split('\\')[:-1] )
    imgFolderName = bgLoc.split('\\')[-1]
    labelFolderName = imgFolderName + '_labels'
    labelFolderLoc = os.path.join( imgFolderParentDir, labelFolderName )

#-------------------------------------------------------------------------------

    # Number of files already existing in the imgSaveLoc is calculated. This 
    # will be used to assign the index to the file while saving.
    nAlreadyExistingFiles = len( os.listdir( imgSaveLoc ) )
    
    bgList, sampleList1, sampleList2, sampleList3 = [], [], [], []
    
    # Creating the images.    
    for i in range( nImgs ):
        # Fill the lists if they are empty.
        # As a sample and a bg is used for creating an image, they are deleted
        # from this list. So if this list gets empty, then it is reinitialized.
        if len( bgList ) == 0:      bgList = os.listdir( bgLoc )
        if len( sampleList1 ) == 0:      sampleList1 = os.listdir( sampleLoc1 )
        if len( sampleList2 ) == 0:      sampleList2 = os.listdir( sampleLoc2 )
        if len( sampleList3 ) == 0:      sampleList3 = os.listdir( sampleLoc3 )
        
        # Select a sample1 at random.
        sampleIdx1 = np.random.randint( len( sampleList1 ) )
        sampleIdx2 = np.random.randint( len( sampleList2 ) )
        sampleIdx3 = np.random.randint( len( sampleList3 ) )
        
        sampleName1 = sampleList1[ sampleIdx1 ]
        sampleName2 = sampleList2[ sampleIdx2 ]
        sampleName3 = sampleList3[ sampleIdx3 ]
        sample1 = cv2.imread( os.path.join( sampleLoc1, sampleName1 ) )
        sample2 = cv2.imread( os.path.join( sampleLoc2, sampleName2 ) )
        sample3 = cv2.imread( os.path.join( sampleLoc3, sampleName3 ) )
        
        className1 = sampleName1.split('_')[0]
        className2 = sampleName2.split('_')[0]
        className3 = sampleName3.split('_')[0]
        
        if maskPresent1:
            # If name of sample is Eosinophil_1.bmp, the name of the 
            # corresponding mask is Eosinophil_1_mask.bmp
            maskName1 = sampleName1[:-4] + '_mask.bmp'
            mask1 = cv2.imread( os.path.join( maskLoc1, maskName1 ) )
        else:
            # If mask is not present then a dummy mask is created which is just
            # a blank array of 255s, of the same type and shape as sample.
            # This makes all future processing easier and also prevents the 
            # check for maskPresent flag every time.
            mask1 = np.ones( sample1.shape ) * 255
            mask1 = np.asarray( mask1, dtype=np.uint8 )
        
        if maskPresent2:
            # If name of sample is Eosinophil_1.bmp, the name of the 
            # corresponding mask is Eosinophil_1_mask.bmp
            maskName2 = sampleName2[:-4] + '_mask.bmp'
            mask2 = cv2.imread( os.path.join( maskLoc2, maskName2 ) )
        else:
            # If mask is not present then a dummy mask is created which is just
            # a blank array of 255s, of the same type and shape as sample.
            # This makes all future processing easier and also prevents the 
            # check for maskPresent flag every time.
            mask2 = np.ones( sample2.shape ) * 255
            mask2 = np.asarray( mask2, dtype=np.uint8 )

        if maskPresent3:
            # If name of sample is Eosinophil_1.bmp, the name of the 
            # corresponding mask is Eosinophil_1_mask.bmp
            maskName3 = sampleName3[:-4] + '_mask.bmp'
            mask3 = cv2.imread( os.path.join( maskLoc3, maskName3 ) )
        else:
            # If mask is not present then a dummy mask is created which is just
            # a blank array of 255s, of the same type and shape as sample.
            # This makes all future processing easier and also prevents the 
            # check for maskPresent flag every time.
            mask3 = np.ones( sample3.shape ) * 255
            mask3 = np.asarray( mask3, dtype=np.uint8 )

        # The bg and sample idxs are determined separately because the number of
        # available samples and bg may be different.
        bgIdx = np.random.randint( len( bgList ) )
        bgName = bgList[ bgIdx ]
        bg = cv2.imread( os.path.join( bgLoc, bgName ) )
        
        # Remove the entry of these samples and bg from the respective lists so 
        # that they are not used again. It will only be used again if all the 
        # existing samples in the lists are used up and the lists become empty.
        sampleList1.pop( sampleIdx1 )
        sampleList2.pop( sampleIdx2 )
        sampleList3.pop( sampleIdx3 )
        bgList.pop( bgIdx )

#-------------------------------------------------------------------------------
        
        # Setting the background.
        
        # It may happen that the bg image is larger than size imgH x imgW.
        # In that case, a imgH x imgW region is cropped out from the bg image.
        bgH, bgW, _ = bg.shape
        
        # Determining the x and y of the top left corner of the region to be
        # cropped out from the bg image.
        bgTlY = np.random.randint( bgH - imgH ) if bgH > imgH else 0
        bgTlX = np.random.randint( bgW - imgW ) if bgW > imgW else 0
        
        # IMPORTANT: The bg image must be larger or equal in size to imgH x imgW.
        newBg1 = bg[ bgTlY : bgTlY + imgH, bgTlX : bgTlX + imgW ]
        
        newBgH, newBgW, _ = newBg1.shape
        
#-------------------------------------------------------------------------------

        # Including the rbc annotations in the label if there are label files 
        # present for these background images.

        # If however the argument includeRbc is false, this function will keep 
        # this dictionary empty.
        if includeRbc:
            # Now those rbc which falls within the region which is cropped for 
            # bg image, are included inside another dictionary.
            # If there are no rbc annotations, then this dictionary will stay empty.
            rbcOnCurrentBg = {}
    
            if os.path.exists( labelFolderLoc ):
                bgLabelName = bgName[:-4] + '.json'
                bgLabelLoc = os.path.join( labelFolderLoc, bgLabelName )
                
                with open( bgLabelLoc, 'r' ) as infoFile:
                    infoDict = json.load( infoFile )
                
                for k, v in infoDict.items():
                    posX, posY = v['posX'], v['posY']
                    if posX >= bgTlX and posX < bgTlX + imgW and \
                       posY >= bgTlY and posY < bgTlY + imgH:
                           rbcOnCurrentBg[k] = v
                           rbcOnCurrentBg[k]['posX'] -= bgTlX
                           rbcOnCurrentBg[k]['posY'] -= bgTlY
                           rbcOnCurrentBg[k]['tlX'] -= bgTlX
                           rbcOnCurrentBg[k]['tlY'] -= bgTlY
                           
        else:
            rbcOnCurrentBg = {}
                       
#-------------------------------------------------------------------------------
            
        # Clip sample1 at the image boundary.

        # If the clipSample flag is True, then the other data augmentation like
        # flipping and rotation is ignored (even if their corresponding flags
        # are True). As this does not make much of a difference.
        
        # All the samples are pasted in the bottom left quadrant and then the 
        # resulting image is rotated to paste the next sample again in the new
        # bottom left quadrant.

        if clipSample1:
            # All clipped samples will have the common name 'partialWBC'.
            className1 = 'partialWBC'

            newSample, newMask = sample1, mask1

            # Whether the clipping will happen at the side or the corner of the
            # image, will be again selected randomly.
            number2 = np.random.randint( 100 )
                
            # The sample will be affixed in a location on the bg, such that 
            # it gets clipped by half. The clipping is always done by half 
            # because, what matters during this clipping is that, the sample 
            # should only be visible by a variable amount inside the image.
            # Now because of the variation of the size of the samples, they 
            # will anyway be visible by variable amount inside the image 
            # even if the percentage of clipping is kept constant. So to 
            # keep things simple the clipping is always done by 50%.
            # Because of the same reason as stated in the previous case, 
            # the clipping at the corners is kept constant at 25% only. The 
            # variability in size of the samples will take care of the rest.
            
            newSampleH, newSampleW, _ = newSample.shape

            tlX = newBgW - int( newSampleW * 0.5 )
            
            if number2 < 60:
                # Clip at the left side of the image.
                tlY = np.random.randint( max( newBgH * 0.5 - newSampleH , 1 ) ) \
                                                + int( newBgH * 0.5 ) 

#-------------------------------------------------------------------------------

            else:    # Clip at the bottom right corner.
                tlY = newBgH - int( newSampleH * 0.5 )
                
            # Fixing the sample onto the background.
            image, posX1, posY1, bboxW1, bboxH1 = fixSampleToBg( newSample, newMask, newBg1, tlX, tlY )
            # Now remove the rbc cells which are overlapped by this wbc cell.
            newRbcOnCurrentBg = { k : v for k, v in rbcOnCurrentBg.items() \
                                  if abs(posX1-v['posX']) > bboxW1/2 or abs(posY1-v['posY']) > bboxH1/2 }
            rbcOnCurrentBg = newRbcOnCurrentBg

#-------------------------------------------------------------------------------
        
        # If the clipSample1 flag is False, then no augmentation is performed.

        else:
            newSample, newMask = sample1, mask1
        
            # x, y of top left corner of the region where sample will be pasted.
            newSampleH, newSampleW, _ = newSample.shape
            tlY = np.random.randint( max( newBgH * 0.5 - newSampleH, 1 ) ) \
                                                + int( newBgH * 0.5 ) 
            tlX = np.random.randint( max( newBgW * 0.5 - newSampleW, 1 ) ) \
                                                + int( newBgW * 0.5 ) 
            
            # Fixing the sample onto the background.
            image, posX1, posY1, bboxW1, bboxH1 = fixSampleToBg( newSample, newMask, newBg1, tlX, tlY )
            # Now remove the rbc cells which are overlapped by this wbc cell.
            newRbcOnCurrentBg = { k : v for k, v in rbcOnCurrentBg.items() \
                                  if abs(posX1-v['posX']) > bboxW1/2 or abs(posY1-v['posY']) > bboxH1/2 }
            rbcOnCurrentBg = newRbcOnCurrentBg

#-------------------------------------------------------------------------------    

        # The new background for sample2 will be the image formed earlier where
        # the sample1 was affixed onto the background.
        newBg2 = image
        newBgH, newBgW, _ = newBg2.shape
        
        # Flip by 90 deg (same as horizontal flip + transpose).
        newBg2 = cv2.transpose( cv2.flip( newBg2, 1 ) )
        posX1, posY1 = posY1, imgW - posX1
        bboxW1, bboxH1 = bboxH1, bboxW1
        # Now modifying the rbc cell locations.
        for k, v in rbcOnCurrentBg.items():
            v['posX'], v['posY'] = v['posY'], imgW - v['posX']
            v['bboxW'], v['bboxH'] = v['bboxH'], v['bboxW']

#-------------------------------------------------------------------------------
        
        if clipSample2:
            # All clipped samples will have the common name 'partialWBC'.
            className2 = 'partialWBC'

            newSample, newMask = sample2, mask2

            # Whether the clipping will happen at the side or the corner of the
            # image, will be again selected randomly.
            number2 = np.random.randint( 100 )
                
            # The sample will be affixed in a location on the bg, such that 
            # it gets clipped by half. The clipping is always done by half 
            # because, what matters during this clipping is that, the sample 
            # should only be visible by a variable amount inside the image.
            # Now because of the variation of the size of the samples, they 
            # will anyway be visible by variable amount inside the image 
            # even if the percentage of clipping is kept constant. So to 
            # keep things simple the clipping is always done by 50%.
            # Because of the same reason as stated in the previous case, 
            # the clipping at the corners is kept constant at 25% only. The 
            # variability in size of the samples will take care of the rest.
            
            newSampleH, newSampleW, _ = newSample.shape

            tlX = newBgW - int( newSampleW * 0.5 )
            
            if number2 < 60:
                # Clip at the left side of the image.
                tlY = np.random.randint( max( newBgH * 0.5 - newSampleH , 1 ) ) \
                                                + int( newBgH * 0.5 ) 

#-------------------------------------------------------------------------------

            else:    # Clip at the bottom right corner.
                tlY = newBgH - int( newSampleH * 0.5 )
                
            # Fixing the sample onto the background.
            image, posX2, posY2, bboxW2, bboxH2 = fixSampleToBg( newSample, newMask, newBg2, tlX, tlY )
            # Now remove the rbc cells which are overlapped by this wbc cell.
            newRbcOnCurrentBg = { k : v for k, v in rbcOnCurrentBg.items() \
                                  if abs(posX2-v['posX']) > bboxW2/2 or abs(posY2-v['posY']) > bboxH2/2 }
            rbcOnCurrentBg = newRbcOnCurrentBg

#-------------------------------------------------------------------------------
        
        # If the clipSample2 flag is False, then no augmentation is performed.

        else:
            newSample, newMask = sample2, mask2
        
            # x, y of top left corner of the region where sample will be pasted.
            newSampleH, newSampleW, _ = newSample.shape
            tlY = np.random.randint( max( newBgH * 0.5 - newSampleH, 1 ) ) \
                                                + int( newBgH * 0.5 ) 
            tlX = np.random.randint( max( newBgW * 0.5 - newSampleW, 1 ) ) \
                                                + int( newBgW * 0.5 ) 
            
            # Fixing the sample onto the background.
            image, posX2, posY2, bboxW2, bboxH2 = fixSampleToBg( newSample, newMask, newBg2, tlX, tlY )
            # Now remove the rbc cells which are overlapped by this wbc cell.
            newRbcOnCurrentBg = { k : v for k, v in rbcOnCurrentBg.items() \
                                  if abs(posX2-v['posX']) > bboxW2/2 or abs(posY2-v['posY']) > bboxH2/2 }
            rbcOnCurrentBg = newRbcOnCurrentBg

#-------------------------------------------------------------------------------    

        # The new background for sample3 will be the image formed earlier where
        # the sample2 was affixed onto the background.
        newBg3 = image
        newBgH, newBgW, _ = newBg3.shape
        
        # Flip by 90 deg (same as horizontal flip + transpose).
        newBg3 = cv2.transpose( cv2.flip( newBg3, 1 ) )
        posX2, posY2 = posY2, imgW - posX2
        bboxW2, bboxH2 = bboxH2, bboxW2
        posX1, posY1 = posY1, imgW - posX1
        bboxW1, bboxH1 = bboxH1, bboxW1
        # Now modifying the rbc cell locations.
        for k, v in rbcOnCurrentBg.items():
            v['posX'], v['posY'] = v['posY'], imgW - v['posX']
            v['bboxW'], v['bboxH'] = v['bboxH'], v['bboxW']

#-------------------------------------------------------------------------------
        
        if clipSample3:
            # All clipped samples will have the common name 'partialWBC'.
            className3 = 'partialWBC'

            newSample, newMask = sample3, mask3

            # Whether the clipping will happen at the side or the corner of the
            # image, will be again selected randomly.
            number2 = np.random.randint( 100 )
                
            # The sample will be affixed in a location on the bg, such that 
            # it gets clipped by half. The clipping is always done by half 
            # because, what matters during this clipping is that, the sample 
            # should only be visible by a variable amount inside the image.
            # Now because of the variation of the size of the samples, they 
            # will anyway be visible by variable amount inside the image 
            # even if the percentage of clipping is kept constant. So to 
            # keep things simple the clipping is always done by 50%.
            # Because of the same reason as stated in the previous case, 
            # the clipping at the corners is kept constant at 25% only. The 
            # variability in size of the samples will take care of the rest.
            
            newSampleH, newSampleW, _ = newSample.shape

            tlX = newBgW - int( newSampleW * 0.5 )
            
            if number2 < 60:
                # Clip at the left side of the image.
                tlY = np.random.randint( max( newBgH * 0.5 - newSampleH , 1 ) ) \
                                                + int( newBgH * 0.5 ) 

#-------------------------------------------------------------------------------

            else:    # Clip at the bottom right corner.
                tlY = newBgH - int( newSampleH * 0.5 )
                
            # Fixing the sample onto the background.
            image, posX3, posY3, bboxW3, bboxH3 = fixSampleToBg( newSample, newMask, newBg3, tlX, tlY )
            # Now remove the rbc cells which are overlapped by this wbc cell.
            newRbcOnCurrentBg = { k : v for k, v in rbcOnCurrentBg.items() \
                                  if abs(posX3-v['posX']) > bboxW3/2 or abs(posY3-v['posY']) > bboxH3/2 }
            rbcOnCurrentBg = newRbcOnCurrentBg

#-------------------------------------------------------------------------------
        
        # If the clipSample3 flag is False, then no augmentation is performed.

        else:
            newSample, newMask = sample3, mask3
        
            # x, y of top left corner of the region where sample will be pasted.
            newSampleH, newSampleW, _ = newSample.shape
            tlY = np.random.randint( max( newBgH * 0.5 - newSampleH, 1 ) ) \
                                                + int( newBgH * 0.5 ) 
            tlX = np.random.randint( max( newBgW * 0.5 - newSampleW, 1 ) ) \
                                                + int( newBgW * 0.5 ) 
            
            # Fixing the sample onto the background.
            image, posX3, posY3, bboxW3, bboxH3 = fixSampleToBg( newSample, newMask, newBg3, tlX, tlY )
            # Now remove the rbc cells which are overlapped by this wbc cell.
            newRbcOnCurrentBg = { k : v for k, v in rbcOnCurrentBg.items() \
                                  if abs(posX3-v['posX']) > bboxW3/2 or abs(posY3-v['posY']) > bboxH3/2 }
            rbcOnCurrentBg = newRbcOnCurrentBg

##-------------------------------------------------------------------------------
#
#        cv2.imshow( 'sample', sample )
#        cv2.imshow( 'newSample', newSample )
#        cv2.imshow( 'mask', mask )
#        cv2.imshow( 'newMask', newMask )
#        cv2.imshow( 'newBg', newBg )
#        cv2.waitKey(0)
#                     
#-------------------------------------------------------------------------------

        # Saving the image.
        idx = nAlreadyExistingFiles + i    # This is the image index.
    
        imgSaveName = className1[:4] + '_' + \
                      className2[:4] + '_' + \
                      className3[:4] + '_' + \
                      saveNameSuffix + '_' + str( idx ) + '.bmp'
                      
        cv2.imwrite( os.path.join( imgSaveLoc, imgSaveName ), image )
        
        # Creating the label json file.
        labelSaveName = className1[:4] + '_' + \
                        className2[:4] + '_' + \
                        className3[:4] + '_' + \
                        saveNameSuffix + '_' + str( idx ) + '.json'
        
        classIdx1 = classNameToIdx[ className1 ]
        classIdx2 = classNameToIdx[ className2 ]
        classIdx3 = classNameToIdx[ className3 ]
        
        infoDict = {}
        
        with open( os.path.join( labelSaveLoc, labelSaveName ), 'w' ) as infoFile:
            
#-------------------------------------------------------------------------------

            posX, posY, bboxW, bboxH = posX1, posY1, bboxW1, bboxH1
            className, classIdx = className1, classIdx1
            sampleLoc, sampleName = sampleLoc1, sampleName1

#-------------------------------------------------------------------------------

            # Make sure the coordinates are inside the boundaries of the image.
            if posX >= imgW:      posX = imgW - 1
            if posX < 0:            posX = 0
            if posY >= imgH:      posY = imgH - 1
            if posY < 0:            posY = 0
            tlX, tlY = posX-bboxW*0.5, posY-bboxH*0.5   # Top left corner.
            brX, brY = posX+bboxW*0.5, posY+bboxH*0.5   # Bottom right corner.
            if tlX < 0:            tlX = 0
            if tlY < 0:            tlY = 0
            if brX >= imgW:      brX = imgW - 1
            if brY >= imgH:      brY = imgH - 1
            bboxW, bboxH = int( brX - tlX ), int( brY - tlY )   # Update box size.
            
#-------------------------------------------------------------------------------

            infoDict[0] = {
                            'className': className, 'classIdx': classIdx, \
                            'posX': int(posX), 'posY': int(posY), \
                            'bboxW': bboxW, 'bboxH': bboxH, \
                            'tlX': int(tlX), 'tlY': int(tlY), \
                            'samplePath': os.path.join( sampleLoc, sampleName ), \
                            'bgPath': os.path.join( bgLoc, bgName ) \
                          }
            
#-------------------------------------------------------------------------------

            posX, posY, bboxW, bboxH = posX2, posY2, bboxW2, bboxH2
            className, classIdx = className2, classIdx2
            sampleLoc, sampleName = sampleLoc2, sampleName2

#-------------------------------------------------------------------------------

            # Make sure the coordinates are inside the boundaries of the image.
            if posX >= imgW:      posX = imgW - 1
            if posX < 0:            posX = 0
            if posY >= imgH:      posY = imgH - 1
            if posY < 0:            posY = 0
            tlX, tlY = posX-bboxW*0.5, posY-bboxH*0.5   # Top left corner.
            brX, brY = posX+bboxW*0.5, posY+bboxH*0.5   # Bottom right corner.
            if tlX < 0:            tlX = 0
            if tlY < 0:            tlY = 0
            if brX >= imgW:      brX = imgW - 1
            if brY >= imgH:      brY = imgH - 1
            bboxW, bboxH = int( brX - tlX ), int( brY - tlY )   # Update box size.
            
#-------------------------------------------------------------------------------

            infoDict[1] = {
                            'className': className, 'classIdx': classIdx, \
                            'posX': int(posX), 'posY': int(posY), \
                            'bboxW': bboxW, 'bboxH': bboxH, \
                            'tlX': int(tlX), 'tlY': int(tlY), \
                            'samplePath': os.path.join( sampleLoc, sampleName ), \
                            'bgPath': os.path.join( bgLoc, bgName ) \
                          }
            
#-------------------------------------------------------------------------------

            posX, posY, bboxW, bboxH = posX3, posY3, bboxW3, bboxH3
            className, classIdx = className3, classIdx3
            sampleLoc, sampleName = sampleLoc3, sampleName3

#-------------------------------------------------------------------------------

            # Make sure the coordinates are inside the boundaries of the image.
            if posX >= imgW:      posX = imgW - 1
            if posX < 0:            posX = 0
            if posY >= imgH:      posY = imgH - 1
            if posY < 0:            posY = 0
            tlX, tlY = posX-bboxW*0.5, posY-bboxH*0.5   # Top left corner.
            brX, brY = posX+bboxW*0.5, posY+bboxH*0.5   # Bottom right corner.
            if tlX < 0:            tlX = 0
            if tlY < 0:            tlY = 0
            if brX >= imgW:      brX = imgW - 1
            if brY >= imgH:      brY = imgH - 1
            bboxW, bboxH = int( brX - tlX ), int( brY - tlY )   # Update box size.
            
#-------------------------------------------------------------------------------

            infoDict[2] = {
                            'className': className, 'classIdx': classIdx, \
                            'posX': int(posX), 'posY': int(posY), \
                            'bboxW': bboxW, 'bboxH': bboxH, \
                            'tlX': int(tlX), 'tlY': int(tlY), \
                            'samplePath': os.path.join( sampleLoc, sampleName ), \
                            'bgPath': os.path.join( bgLoc, bgName ) \
                          }
            
#-------------------------------------------------------------------------------

            # Now recording the rbc cells into the infoDict.
            nWbc = len( infoDict )      # Number of wbc cell records in infoDict.
            print( len(rbcOnCurrentBg) )
            
            for r, (k, v) in enumerate( rbcOnCurrentBg.items() ):
                # Creating key for the rbc cell records. 
                # This makes sure that they are different from the keys of the 
                # wbc cell records, else they may overlap the wbc record.
                index = r + nWbc
                
#-------------------------------------------------------------------------------

                posX, posY, bboxW, bboxH = v['posX'], v['posY'], v['bboxW'], v['bboxH']
                classNameRbc, classIdxRbc = v['className'], v['classIdx']
                
#-------------------------------------------------------------------------------

                # Make sure the coordinates are inside the boundaries of the image.
                if posX >= imgW:      posX = imgW - 1
                if posX < 0:            posX = 0
                if posY >= imgH:      posY = imgH - 1
                if posY < 0:            posY = 0
                tlX, tlY = posX-bboxW*0.5, posY-bboxH*0.5   # Top left corner.
                brX, brY = posX+bboxW*0.5, posY+bboxH*0.5   # Bottom right corner.
                if tlX < 0:            tlX = 0
                if tlY < 0:            tlY = 0
                if brX >= imgW:      brX = imgW - 1
                if brY >= imgH:      brY = imgH - 1
                bboxW, bboxH = int( brX - tlX ), int( brY - tlY )   # Update box size.
            
#-------------------------------------------------------------------------------

                infoDict[ index ] = {
                                        'className': classNameRbc, 'classIdx': classIdxRbc, \
                                        'posX': int(posX), 'posY': int(posY), \
                                        'bboxW': bboxW, 'bboxH': bboxH, \
                                        'tlX': int(tlX), 'tlY': int(tlY), \
                                        'samplePath': None, \
                                        'bgPath': os.path.join( bgLoc, bgName ) \
                                    }
                
            json.dump( infoDict, infoFile, indent=4, separators=(',', ': ') )

#-------------------------------------------------------------------------------
        
        for k, v in infoDict.items():
            cv2.circle( image, (v['posX'], v['posY']), 2, (0,255,0), 2 )
            if v['className'] != '_':
                cv2.rectangle( image, (v['tlX'], v['tlY']), (v['tlX']+v['bboxW'], \
                                       v['tlY']+v['bboxH']), (0,255,0), 2 )
        cv2.imshow( 'image', image )
        cv2.waitKey(30)

#===============================================================================

def multiInstance( sampleLoc1=None, sampleLoc2=None, sampleLoc3=None, \
                   sampleLoc4=None, sampleLoc5=None, \
                   maskLoc1=None, maskLoc2=None, maskLoc3=None, \
                   maskLoc4=None, maskLoc5=None, bgLoc=None, \
                   imgSaveLoc=None, labelSaveLoc=None, nImgs=None, imgH=None, \
                   imgW=None, saveNameSuffix=None, includeRbc=False ):
    '''
    This function creates images where an object from each of the sampleLoc folders
    are randomly selected and pasted on a background seleced from the bgLoc folder. 
    The maskLoc folders holds the masks for the sample of sampleLoc folder, 
    but these are optional. If there are no maskLoc provided, then the 
    corresponding samples are just pasted as they are, otherwise the corresponding 
    mask is used while pasting the sample.
    The samples and backgrounds are selected randomly from the available 
    collection in their respective locations. Total number of images created is 
    nImgs. These images are saved in the imgSaveLoc.
    The labels of the corresponding images are also created as json files in 
    the labelSaveLoc.
    imgH and imgW defines the size of the image to be created.
    The saveNameSuffix is a string, that will be appended to the name of the 
    image file while saving. This is important to identify the dataset from 
    where the image has been synthesized.
    IMPORTANT: The background images in this case has to be at least 600 x 400.
    The function also checks if there are already some rbc annotations present 
    on the background or not. If so then it includes them in the labels.
    '''
    
    if sampleLoc1 is None or sampleLoc2 is None or sampleLoc3 is None \
       or sampleLoc4 is None or sampleLoc5 is None or bgLoc is None \
       or imgSaveLoc is None or labelSaveLoc is None or nImgs is None \
       or imgH is None or imgW is None or saveNameSuffix is None:
           print( '\nERROR: one or more input arguments missing ' \
                  'in multiInstance. Aborting.\n' )
           sys.exit()
    
    # Flag indicating mask present.
    maskPresent1 = False if maskLoc1 is None else True
    maskPresent2 = False if maskLoc2 is None else True
    maskPresent3 = False if maskLoc3 is None else True
    maskPresent4 = False if maskLoc4 is None else True
    maskPresent5 = False if maskLoc5 is None else True
    
#-------------------------------------------------------------------------------
        
    # Checking if there is any label file for the bg images present or not. 
    # These files will include rbc annotations, if present.
    imgFolderParentDir = '\\'.join( bgLoc.split('\\')[:-1] )
    imgFolderName = bgLoc.split('\\')[-1]
    labelFolderName = imgFolderName + '_labels'
    labelFolderLoc = os.path.join( imgFolderParentDir, labelFolderName )

#-------------------------------------------------------------------------------

    # Number of files already existing in the imgSaveLoc is calculated. This 
    # will be used to assign the index to the file while saving.
    nAlreadyExistingFiles = len( os.listdir( imgSaveLoc ) )
    
    bgList, sampleList1, sampleList2, sampleList3, sampleList4, sampleList5 = \
                                                    [], [], [], [], [], []
    
    # Creating the images.    
    for i in range( nImgs ):
        # Fill the lists if they are empty.
        # As a sample and a bg is used for creating an image, they are deleted
        # from this list. So if this list gets empty, then it is reinitialized.
        if len( bgList ) == 0:      bgList = os.listdir( bgLoc )
        if len( sampleList1 ) == 0:      sampleList1 = os.listdir( sampleLoc1 )
        if len( sampleList2 ) == 0:      sampleList2 = os.listdir( sampleLoc2 )
        if len( sampleList3 ) == 0:      sampleList3 = os.listdir( sampleLoc3 )
        if len( sampleList4 ) == 0:      sampleList4 = os.listdir( sampleLoc4 )
        if len( sampleList5 ) == 0:      sampleList5 = os.listdir( sampleLoc5 )
        
        # Select a sample1 at random, either 1 or 2 samples will be selected.
        # The number of samples (1 or 2) is also selected at random.
        sampleIdxList1 = np.random.randint( len( sampleList1 ), size=np.random.randint(1,3) )
        sampleIdxList2 = np.random.randint( len( sampleList2 ), size=np.random.randint(1,3) )
        sampleIdxList3 = np.random.randint( len( sampleList3 ), size=np.random.randint(1,3) )
        sampleIdxList4 = np.random.randint( len( sampleList4 ), size=np.random.randint(1,3) )
        sampleIdxList5 = np.random.randint( len( sampleList5 ), size=np.random.randint(1,3) )
        
        # These lists may have repeatation of indexes. So those duplicate indexes
        # are removed by converting the lists into sets.
        sampleIdxList1 = list( set( sampleIdxList1.tolist() ) )
        sampleIdxList2 = list( set( sampleIdxList2.tolist() ) )
        sampleIdxList3 = list( set( sampleIdxList3.tolist() ) )
        sampleIdxList4 = list( set( sampleIdxList4.tolist() ) )
        sampleIdxList5 = list( set( sampleIdxList5.tolist() ) )

        # Now combining all the sample lists.
        sampleList, maskList, classNameList = [], [], []
        
        sampleNameList1 = []
        for s in sampleIdxList1:
            sampleName = sampleList1[s]
            className = sampleName.split('_')[0]
            sample = cv2.imread( os.path.join( sampleLoc1, sampleName ) )
            sampleNameList1.append( sampleName )
            classNameList.append( className )
            sampleList.append( sample )
            
            if maskPresent1:
                # If name of sample is Eosinophil_1.bmp, the name of the 
                # corresponding mask is Eosinophil_1_mask.bmp
                maskName = sampleName[:-4] + '_mask.bmp'
                mask = cv2.imread( os.path.join( maskLoc1, maskName ) )
            else:
                # If mask is not present then a dummy mask is created which is just
                # a blank array of 255s, of the same type and shape as sample.
                # This makes all future processing easier and also prevents the 
                # check for maskPresent flag every time.
                mask = np.ones( sample.shape ) * 255
                mask = np.asarray( mask, dtype=np.uint8 )
            maskList.append( mask )
            
        # Remove the entry of these samples and bg from the respective lists so 
        # that they are not used again. It will only be used again if all the 
        # existing samples in the lists are used up and the lists become empty.
        for n in sampleNameList1:                sampleList1.remove( n )

        sampleNameList2 = []
        for s in sampleIdxList2:
            sampleName = sampleList2[s]
            className = sampleName.split('_')[0]
            sample = cv2.imread( os.path.join( sampleLoc2, sampleName ) )
            sampleNameList2.append( sampleName )
            classNameList.append( className )
            sampleList.append( sample )
            
            if maskPresent2:
                # If name of sample is Eosinophil_1.bmp, the name of the 
                # corresponding mask is Eosinophil_1_mask.bmp
                maskName = sampleName[:-4] + '_mask.bmp'
                mask = cv2.imread( os.path.join( maskLoc2, maskName ) )
            else:
                # If mask is not present then a dummy mask is created which is just
                # a blank array of 255s, of the same type and shape as sample.
                # This makes all future processing easier and also prevents the 
                # check for maskPresent flag every time.
                mask = np.ones( sample.shape ) * 255
                mask = np.asarray( mask, dtype=np.uint8 )
            maskList.append( mask )
            
        # Remove the entry of these samples and bg from the respective lists so 
        # that they are not used again. It will only be used again if all the 
        # existing samples in the lists are used up and the lists become empty.
        for n in sampleNameList2:                sampleList2.remove( n )
        
        sampleNameList3 = []
        for s in sampleIdxList3:
            sampleName = sampleList3[s]
            className = sampleName.split('_')[0]
            sample = cv2.imread( os.path.join( sampleLoc3, sampleName ) )
            sampleNameList3.append( sampleName )
            classNameList.append( className )
            sampleList.append( sample )
        
            if maskPresent3:
                # If name of sample is Eosinophil_1.bmp, the name of the 
                # corresponding mask is Eosinophil_1_mask.bmp
                maskName = sampleName[:-4] + '_mask.bmp'
                mask = cv2.imread( os.path.join( maskLoc3, maskName ) )
            else:
                # If mask is not present then a dummy mask is created which is just
                # a blank array of 255s, of the same type and shape as sample.
                # This makes all future processing easier and also prevents the 
                # check for maskPresent flag every time.
                mask = np.ones( sample.shape ) * 255
                mask = np.asarray( mask, dtype=np.uint8 )
            maskList.append( mask )

        # Remove the entry of these samples and bg from the respective lists so 
        # that they are not used again. It will only be used again if all the 
        # existing samples in the lists are used up and the lists become empty.
        for n in sampleNameList3:                sampleList3.remove( n )
        
        sampleNameList4 = []
        for s in sampleIdxList4:
            sampleName = sampleList4[s]
            className = sampleName.split('_')[0]
            sample = cv2.imread( os.path.join( sampleLoc4, sampleName ) )
            sampleNameList4.append( sampleName )
            classNameList.append( className )
            sampleList.append( sample )

            if maskPresent4:
                # If name of sample is Eosinophil_1.bmp, the name of the 
                # corresponding mask is Eosinophil_1_mask.bmp
                maskName = sampleName[:-4] + '_mask.bmp'
                mask = cv2.imread( os.path.join( maskLoc4, maskName ) )
            else:
                # If mask is not present then a dummy mask is created which is just
                # a blank array of 255s, of the same type and shape as sample.
                # This makes all future processing easier and also prevents the 
                # check for maskPresent flag every time.
                mask = np.ones( sample.shape ) * 255
                mask = np.asarray( mask, dtype=np.uint8 )
            maskList.append( mask )
            
        # Remove the entry of these samples and bg from the respective lists so 
        # that they are not used again. It will only be used again if all the 
        # existing samples in the lists are used up and the lists become empty.
        for n in sampleNameList4:                sampleList4.remove( n )
        
        sampleNameList5 = []
        for s in sampleIdxList5:
            sampleName = sampleList5[s]
            className = sampleName.split('_')[0]
            sample = cv2.imread( os.path.join( sampleLoc5, sampleName ) )
            sampleNameList5.append( sampleName )
            classNameList.append( className )
            sampleList.append( sample )

            if maskPresent5:
                # If name of sample is Eosinophil_1.bmp, the name of the 
                # corresponding mask is Eosinophil_1_mask.bmp
                maskName = sampleName[:-4] + '_mask.bmp'
                mask = cv2.imread( os.path.join( maskLoc5, maskName ) )
            else:
                # If mask is not present then a dummy mask is created which is just
                # a blank array of 255s, of the same type and shape as sample.
                # This makes all future processing easier and also prevents the 
                # check for maskPresent flag every time.
                mask = np.ones( sample.shape ) * 255
                mask = np.asarray( mask, dtype=np.uint8 )
            maskList.append( mask )
            
        # Remove the entry of these samples and bg from the respective lists so 
        # that they are not used again. It will only be used again if all the 
        # existing samples in the lists are used up and the lists become empty.
        for n in sampleNameList5:                sampleList5.remove( n )
        
        # The bg and sample idxs are determined separately because the number of
        # available samples and bg may be different.
        bgIdx = np.random.randint( len( bgList ) )
        bgName = bgList[ bgIdx ]
        bg = cv2.imread( os.path.join( bgLoc, bgName ) )
        
        # Remove the entry of these samples and bg from the respective lists so 
        # that they are not used again. It will only be used again if all the 
        # existing samples in the lists are used up and the lists become empty.
        bgList.pop( bgIdx )
        
        sampleNameList = sampleNameList1 + sampleNameList2 + sampleNameList3 \
                                         + sampleNameList4 + sampleNameList5
                                         
        infoDict = {}       # Empty dictionary to store the label information.

#-------------------------------------------------------------------------------
        
        # Setting the background.
        
        # The whole bg is divided into a 2 row, 3 col grid.
        bgH, bgW, _ = bg.shape
        
        bg1 = bg[ 0 : int(bgH/2), 0 : int(bgW/3), : ]
        bg2 = bg[ 0 : int(bgH/2), int(bgW/3) : int(2*bgW/3), : ]
        bg3 = bg[ 0 : int(bgH/2), int(2*bgW/3) : bgW, : ]
        bg4 = bg[ int(bgH/2) : bgH, 0 : int(bgW/3), : ]
        bg5 = bg[ int(bgH/2) : bgH, int(bgW/3) : int(2*bgW/3), : ]
        bg6 = bg[ int(bgH/2) : bgH, int(2*bgW/3) : bgW, : ]
        
        # The offset lists will be used to calculate the true location of the 
        # samples in the complete final image.
        tlXoffsetList = [ 0, int(bgW/3), int(2*bgW/3), 0, int(bgW/3), int(2*bgW/3) ]
        tlYoffsetList = [ 0, 0, 0, int(bgH/2), int(bgH/2), int(bgH/2) ]
        
        gridCellList = [ bg1, bg2, bg3, bg4, bg5, bg6 ]
        nGridCells = len( gridCellList )
        gcIdList = list( range( nGridCells ) )    # Will be used to store 
        # the image in proper location in the imageList.
        
        sampleLabelIdx = 0
        
        # Now picking any bg from gridCellList at random and pasting the sample 
        # in that. The resulting images are saved in another list called imageList.
        # This imageList is initialized to the gridCellList before anything is
        # pasted.
        imageList = copy.deepcopy( gridCellList )
        
#-------------------------------------------------------------------------------
        
        for b in range( nGridCells ):
            # As a sample and a bg is used for creating an image, they are deleted
            # from their lists. So the number of remaining bg grid cells and 
            # samples need to be updated.
            nRemainingGridCells = len( gridCellList )
            nRemainingSamples = len( sampleList )

            if nRemainingSamples <= nRemainingGridCells and nRemainingSamples > 0:
                # If we have more or equal number of grid cells and samples left, 
                # then one sample is pasted in every grid cell. Some grid cells
                # are kept blank if the number of samples remaining is less
                # than the number of grid cells.
                gcIdx = np.random.randint( nRemainingGridCells )
                newBg = gridCellList[ gcIdx ]
                newBgId = gcIdList[ gcIdx ]
                sampleIdx = np.random.randint( nRemainingSamples )
                sample = sampleList[ sampleIdx ]
                className = classNameList[ sampleIdx ]
                sampleName = sampleNameList[ sampleIdx ]
                mask = maskList[ sampleIdx ]
        
                # x, y of top left corner of the region where sample will be pasted.
                sampleH, sampleW, _ = sample.shape
                newBgH, newBgW, _ = newBg.shape
                tlY = np.random.randint( newBgH - sampleH )
                tlX = np.random.randint( newBgW - sampleW )
        
                # Fixing the sample onto the background.
                image, posX, posY, bboxW, bboxH = fixSampleToBg( sample, mask, \
                                                                 newBg, tlX, tlY )

#-------------------------------------------------------------------------------

                # Save the information for the labels in a dictionary.
                imgH, imgW, _ = image.shape
                tlXoffset = tlXoffsetList[ newBgId ]    # x offset for this bg.
                tlYoffset = tlYoffsetList[ newBgId ]    # y offset for this bg.

                infoDict[ sampleLabelIdx ] = \
                             {
                                'className': className, \
                                'classIdx': classNameToIdx[ className ], \
                                'posX': int(posX) + tlXoffset, \
                                'posY': int(posY) + tlYoffset, \
                                'bboxW': bboxW, 'bboxH': bboxH, \
                                'tlX': int(posX-bboxW*0.5) + tlXoffset, \
                                'tlY': int(posY-bboxH*0.5) + tlYoffset, \
                                'samplePath': sampleName, \
                                'bgPath': os.path.join( bgLoc, bgName ) \
                              }
                sampleLabelIdx += 1
                
#-------------------------------------------------------------------------------

                # Store the image.
                imageList[ newBgId ] = image
                
                # Remove the sample and bg that is used, from their lists.
                gridCellList.pop( gcIdx )
                gcIdList.pop( gcIdx )
                sampleList.pop( sampleIdx )
                classNameList.pop( sampleIdx )
                sampleNameList.pop( sampleIdx )
                maskList.pop( sampleIdx )
                
#-------------------------------------------------------------------------------

            elif nRemainingSamples > nRemainingGridCells:
                # If we have more number of samples than number of grid cells left, 
                # then two samples are pasted in every grid cell. This is done
                # until the remaining number of samples becomes less than or 
                # equal to the number of samples remaining.
                gcIdx = np.random.randint( nRemainingGridCells )
                newBg = gridCellList[ gcIdx ]
                newBgId = gcIdList[ gcIdx ]
                
                sampleIdx = np.random.randint( nRemainingSamples )
                sample = sampleList[ sampleIdx ]
                className = classNameList[ sampleIdx ]
                sampleName = sampleNameList[ sampleIdx ]
                mask = maskList[ sampleIdx ]
        
                # x, y of top left corner of the region where sample will be pasted.
                sampleH, sampleW, _ = sample.shape
                newBgH, newBgW, _ = newBg.shape
                
                # Fixing the sample onto the background.
                # The first sample is always pasted in the top left quadrant.
                tlX = np.random.randint( max( newBgW * 0.5 - sampleW , 1 ) )
                tlY = np.random.randint( max( newBgH * 0.5 - sampleH , 1 ) )
                image, posX, posY, bboxW, bboxH = fixSampleToBg( sample, mask, \
                                                                 newBg, tlX, tlY )

#-------------------------------------------------------------------------------

                # Save the information for the labels in a dictionary.
                imgH, imgW, _ = image.shape
                tlXoffset = tlXoffsetList[ newBgId ]    # x offset for this bg.
                tlYoffset = tlYoffsetList[ newBgId ]    # y offset for this bg.
                infoDict[ sampleLabelIdx ] = \
                             {
                                'className': className, \
                                'classIdx': classNameToIdx[ className ], \
                                'posX': int(posX) + tlXoffset, \
                                'posY': int(posY) + tlYoffset, \
                                'bboxW': bboxW, 'bboxH': bboxH, \
                                'tlX': int(posX-bboxW*0.5) + tlXoffset, \
                                'tlY': int(posY-bboxH*0.5) + tlYoffset, \
                                'samplePath': sampleName, \
                                'bgPath': os.path.join( bgLoc, bgName ) \
                              }
                sampleLabelIdx += 1
                
#-------------------------------------------------------------------------------

                # Remove the sample from the lists.
                sampleList.pop( sampleIdx )
                classNameList.pop( sampleIdx )
                sampleNameList.pop( sampleIdx )
                maskList.pop( sampleIdx )

#-------------------------------------------------------------------------------

                # The image from the last operation will be the background of the
                # second sample.
                newBg = copy.deepcopy( image )                

                nRemainingSamples = len( sampleList )
                sampleIdx = np.random.randint( nRemainingSamples )
                sample = sampleList[ sampleIdx ]
                className = classNameList[ sampleIdx ]
                sampleName = sampleNameList[ sampleIdx ]
                mask = maskList[ sampleIdx ]

                # The quadrant of bg where the second sample will be pasted is 
                # decided at random with equal probability.
                number = np.random.randint( 100 )
                if number < 33:     # Top right quadrant.
                    tlX = np.random.randint( max( newBgW * 0.5 - sampleW , 1 ) ) \
                                                    + int( newBgW * 0.5 )
                    tlY = np.random.randint( max( newBgH * 0.5 - sampleH , 1 ) )
                    # Fixing the sample onto the background.
                    image, posX, posY, bboxW, bboxH = fixSampleToBg( sample, mask, \
                                                                     newBg, tlX, tlY )
                elif number >= 33 and number < 66:     # Bottom left quadrant.
                    tlX = np.random.randint( max( newBgW * 0.5 - sampleW , 1 ) )
                    tlY = np.random.randint( max( newBgH * 0.5 - sampleH , 1 ) ) \
                                                    + int( newBgH * 0.5 )
                    # Fixing the sample onto the background.
                    image, posX, posY, bboxW, bboxH = fixSampleToBg( sample, mask, \
                                                                     newBg, tlX, tlY )
                elif number >= 66:     # Bottom right quadrant.
                    tlX = np.random.randint( max( newBgW * 0.5 - sampleW , 1 ) ) \
                                                    + int( newBgW * 0.5 )
                    tlY = np.random.randint( max( newBgH * 0.5 - sampleH , 1 ) ) \
                                                    + int( newBgH * 0.5 )
                    # Fixing the sample onto the background.
                    image, posX, posY, bboxW, bboxH = fixSampleToBg( sample, mask, \
                                                                     newBg, tlX, tlY )                

#-------------------------------------------------------------------------------

                # Save the information for the labels in a dictionary.
                imgH, imgW, _ = image.shape
                tlXoffset = tlXoffsetList[ newBgId ]    # x offset for this bg.
                tlYoffset = tlYoffsetList[ newBgId ]    # y offset for this bg.
                infoDict[ sampleLabelIdx ] = \
                             {
                                'className': className, \
                                'classIdx': classNameToIdx[ className ], \
                                'posX': int(posX) + tlXoffset, \
                                'posY': int(posY) + tlYoffset, \
                                'bboxW': bboxW, 'bboxH': bboxH, \
                                'tlX': int(posX-bboxW*0.5) + tlXoffset, \
                                'tlY': int(posY-bboxH*0.5) + tlYoffset, \
                                'samplePath': sampleName, \
                                'bgPath': os.path.join( bgLoc, bgName ) \
                              }
                sampleLabelIdx += 1
                
#-------------------------------------------------------------------------------

                # Store the image.
                imageList[ newBgId ] = image

                # Remove the sample from the lists.
                sampleList.pop( sampleIdx )
                classNameList.pop( sampleIdx )
                sampleNameList.pop( sampleIdx )
                maskList.pop( sampleIdx )

                # Remove the bg that is used, from the lists.
                gridCellList.pop( gcIdx )
                gcIdList.pop( gcIdx )

#-------------------------------------------------------------------------------    

        # Stitching the images back together.
        image1 = np.hstack( ( imageList[0], imageList[1], imageList[2] ) )
        image2 = np.hstack( ( imageList[3], imageList[4], imageList[5] ) )
        image = np.vstack( ( image1, image2 ) )        

#-------------------------------------------------------------------------------

        # Saving the image.
        idx = nAlreadyExistingFiles + i    # This is the image index.
    
        imgSaveName = 'multi_sample' + '_' + str( idx ) + '.bmp'
                      
        cv2.imwrite( os.path.join( imgSaveLoc, imgSaveName ), image )
        
        # Creating the label json file.
        labelSaveName = 'multi_sample' + '_' + str( idx ) + '.json'
        
        with open( os.path.join( labelSaveLoc, labelSaveName ), 'w' ) as infoFile:
            json.dump( infoDict, infoFile, indent=4, separators=(',', ': ') )

#-------------------------------------------------------------------------------
            
        imgH, imgW, _ = image.shape
        for k, v in infoDict.items():
            posX, posY, bboxW, bboxH = v['posX'], v['posY'], v['bboxW'], v['bboxH']
            className, tlX, tlY = v['className'], v['tlX'], v['tlY']
            cx, cy = int(posX), int(posY)
#            cv2.circle( image, (int(posX), int(posY)), 2, (0,255,255), 2 )
#            cv2.rectangle( image, ( int(posX-bboxW*0.5), int(posY-bboxH*0.5) ), \
#                                  ( int(posX+bboxW*0.5), int(posY+bboxH*0.5) ), \
#                                  (0,255,255), 2 )
            ly = tlY-10 if tlY-20 > 0 else tlY+bboxH + 20
            lx = tlX-10 if tlX-20 > 0 else tlX+bboxW + 20
            cv2.putText( image, className.split('_')[0].upper(), (lx, ly), \
                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA )
            cv2.arrowedLine( image, (lx+5, ly+5), (cx, cy), (0,0,0), 2 )

        if not os.path.exists( os.path.join( imgSaveLoc, 'images_with_names' ) ):
            os.makedirs( os.path.join( imgSaveLoc, 'images_with_names' ) )
            
        cv2.imwrite( os.path.join( imgSaveLoc, 'images_with_names', imgSaveName ), image )

        cv2.imshow( 'image', image )
        cv2.waitKey(30)

#===============================================================================
        
def doubleInstance( sampleLoc1=None, sampleLoc2=None, maskLoc1=None, \
                    maskLoc2=None, bgLoc=None, imgSaveLoc=None, labelSaveLoc=None, \
                    nImgs=None, imgH=None, imgW=None,
                    saveNameSuffix=None, do90degFlips=False, \
                    doHoriFlip=False, doVertFlip=False, doRandomRot=False, \
                    clipSample1=False, clipSample2=False, includeRbc=False, \
                    createSegmentLabelImg=False, segmentSaveLoc=None ):
    '''
    This function creates images where an object from each of the folders 
    sampleLoc1 and sampleLoc2 are randomly selected and pasted on a background
    seleced from the bgLoc folder. So there are two instances of wbc in the same 
    image.
    The maskLoc1 and maskLoc2 holds the masks for the sample of sampleLoc1 and 
    sampleLoc2 respectively, but these are optional. If there
    are no maskLoc1 or maskLoc2 provided, then the corresponding samples are 
    just pasted as they are, otherwise the corresponding mask is used while 
    pasting the sample.
    The samples and backgrounds are selected randomly from the available 
    collection in their respective locations. Total number of images created is 
    nImgs. These images are saved in the imgSaveLoc.
    The labels of the corresponding images are also created as json files in 
    the labelSaveLoc.
    imgH and imgW defines the size of the image to be created.
    The saveNameSuffix is a string, that will be appended to the name of the 
    image file while saving. This is important to identify the dataset from 
    where the image has been synthesized.
    The flag do90degFlips indicates whether the sample should undergo rotations
    by multiples of 90 deg (randomly), while getting affixed on the bg image.
    The flag doRandomRot indicates whether the sample should undergo rotations
    by random angles, while getting affixed on the bg image.
    Flags doHoriFlip and doVertFlip indicates if the sample should be flipped 
    horizontally or vertically (randomly) before getting affixed on bg image.
    The clipSample1 (or clipSample2) flag indicates if there will be any 
    clipping of the sample1 (or sample2) when it is affixed on the bg.
    The function also checks if there are already some rbc annotations present 
    on the background or not. If so then it includes them in the labels.
    The createSegmentLabelImg indicates if a semantic segmentation label image 
    has to be created as well. The colors of the segments for different objects 
    are mentioned in the global variables. Segment save location is also provided.
    '''
    
    if sampleLoc1 is None or sampleLoc2 is None or bgLoc is None \
       or imgSaveLoc is None or labelSaveLoc is None or nImgs is None \
       or imgH is None or imgW is None or saveNameSuffix is None:
           print( '\nERROR: one or more input arguments missing ' \
                  'in doubleInstance. Aborting.\n' )
           sys.exit()
    
    if createSegmentLabelImg:
        if segmentSaveLoc is None:
            print( '\nERROR: one or more input arguments missing ' \
                   'in doubleInstance for segments. Aborting.\n' )
            sys.exit()
    
    # Flag indicating mask present.
    maskPresent1 = False if maskLoc1 is None else True
    maskPresent2 = False if maskLoc2 is None else True

#-------------------------------------------------------------------------------
        
    # Checking if there is any label file for the bg images present or not. 
    # These files will include rbc annotations, if present.
    imgFolderParentDir = '\\'.join( bgLoc.split('\\')[:-1] )
    imgFolderName = bgLoc.split('\\')[-1]
    labelFolderName = imgFolderName + '_labels'
    labelFolderLoc = os.path.join( imgFolderParentDir, labelFolderName )
    
    if createSegmentLabelImg:
        bgSegmentFolderName = imgFolderName + '_segments'
        bgSegmentFolderLoc = os.path.join( imgFolderParentDir, bgSegmentFolderName )

#-------------------------------------------------------------------------------

    # Number of files already existing in the imgSaveLoc is calculated. This 
    # will be used to assign the index to the file while saving.
    nAlreadyExistingFiles = len( os.listdir( imgSaveLoc ) )
    
    bgList, sampleList1, sampleList2 = [], [], []
    
    # Creating the images.    
    for i in range( nImgs ):
        # Fill the lists if they are empty.
        # As a sample and a bg is used for creating an image, they are deleted
        # from this list. So if this list gets empty, then it is reinitialized.
        if len( bgList ) == 0:      bgList = os.listdir( bgLoc )
        if len( sampleList1 ) == 0:      sampleList1 = os.listdir( sampleLoc1 )
        if len( sampleList2 ) == 0:      sampleList2 = os.listdir( sampleLoc2 )
        
        # Select a sample1 at random.
        sampleIdx1 = np.random.randint( len( sampleList1 ) )
        sampleIdx2 = np.random.randint( len( sampleList2 ) )
        
        sampleName1 = sampleList1[ sampleIdx1 ]
        sampleName2 = sampleList2[ sampleIdx2 ]
        sample1 = cv2.imread( os.path.join( sampleLoc1, sampleName1 ) )
        sample2 = cv2.imread( os.path.join( sampleLoc2, sampleName2 ) )
                
        className1 = sampleName1.split('_')[0]
        className2 = sampleName2.split('_')[0]
        
        if maskPresent1:
            # If name of sample is Eosinophil_1.bmp, the name of the 
            # corresponding mask is Eosinophil_1_mask.bmp
            maskName1 = sampleName1[:-4] + '_mask.bmp'
            mask1 = cv2.imread( os.path.join( maskLoc1, maskName1 ) )
        else:
            # If mask is not present then a dummy mask is created which is just
            # a blank array of 255s, of the same type and shape as sample.
            # This makes all future processing easier and also prevents the 
            # check for maskPresent flag every time.
            mask1 = np.ones( sample1.shape ) * 255
            mask1 = np.asarray( mask1, dtype=np.uint8 )
        
        if maskPresent2:
            # If name of sample is Eosinophil_1.bmp, the name of the 
            # corresponding mask is Eosinophil_1_mask.bmp
            maskName2 = sampleName2[:-4] + '_mask.bmp'
            mask2 = cv2.imread( os.path.join( maskLoc2, maskName2 ) )
        else:
            # If mask is not present then a dummy mask is created which is just
            # a blank array of 255s, of the same type and shape as sample.
            # This makes all future processing easier and also prevents the 
            # check for maskPresent flag every time.
            mask2 = np.ones( sample2.shape ) * 255
            mask2 = np.asarray( mask2, dtype=np.uint8 )

        # The bg and sample idxs are determined separately because the number of
        # available samples and bg may be different.
        bgIdx = np.random.randint( len( bgList ) )
        bgName = bgList[ bgIdx ]
        bg = cv2.imread( os.path.join( bgLoc, bgName ) )
        
        # Remove the entry of these samples and bg from the respective lists so 
        # that they are not used again. It will only be used again if all the 
        # existing samples in the lists are used up and the lists become empty.
        sampleList1.pop( sampleIdx1 )
        sampleList2.pop( sampleIdx2 )
        bgList.pop( bgIdx )

#-------------------------------------------------------------------------------
        
        # Setting the background.
        
        # It may happen that the bg image is larger than size imgH x imgW.
        # In that case, a imgH x imgW region is cropped out from the bg image.
        bgH, bgW, _ = bg.shape
        
        # Determining the x and y of the top left corner of the region to be
        # cropped out from the bg image.
        bgTlY = np.random.randint( bgH - imgH ) if bgH > imgH else 0
        bgTlX = np.random.randint( bgW - imgW ) if bgW > imgW else 0
        
        # IMPORTANT: The bg image must be larger or equal in size to imgH x imgW.
        newBg1 = bg[ bgTlY : bgTlY + imgH, bgTlX : bgTlX + imgW ]
        
        newBgH, newBgW, _ = newBg1.shape
        
#-------------------------------------------------------------------------------

        # Also doing the same processing for the segmented image label.
        if createSegmentLabelImg:
            bgSegName = 'seg_' + '_'.join( bgName.split('_')[1:] )
            bgSegImg = cv2.imread( os.path.join( bgSegmentFolderLoc, bgSegName ) )
            
            newBgSegImg1 = bgSegImg[ bgTlY : bgTlY + imgH, bgTlX : bgTlX + imgW ]
            
#-------------------------------------------------------------------------------

        # Including the rbc annotations in the label if there are label files 
        # present for these background images.

        # If however the argument includeRbc is false, this function will keep 
        # this dictionary empty.
        if includeRbc:
            # Now those rbc which falls within the region which is cropped for 
            # bg image, are included inside another dictionary.
            # If there are no rbc annotations, then this dictionary will stay empty.
            rbcOnCurrentBg = {}
    
            if os.path.exists( labelFolderLoc ):
                bgLabelName = bgName[:-4] + '.json'
                bgLabelLoc = os.path.join( labelFolderLoc, bgLabelName )
                
                with open( bgLabelLoc, 'r' ) as infoFile:
                    infoDict = json.load( infoFile )
                
                for k, v in infoDict.items():
                    posX, posY = v['posX'], v['posY']
                    if posX >= bgTlX and posX < bgTlX + imgW and \
                       posY >= bgTlY and posY < bgTlY + imgH:
                           rbcOnCurrentBg[k] = v
                           rbcOnCurrentBg[k]['posX'] -= bgTlX
                           rbcOnCurrentBg[k]['posY'] -= bgTlY
                           rbcOnCurrentBg[k]['tlX'] -= bgTlX
                           rbcOnCurrentBg[k]['tlY'] -= bgTlY
                           
        else:
            rbcOnCurrentBg = {}
                       
#-------------------------------------------------------------------------------

        # Clip sample1 at the image boundary.

        # If the clipSample flag is True, then the other data augmentation like
        # flipping and rotation is ignored (even if their corresponding flags
        # are True). As this does not make much of a difference.

        if clipSample1:
            # All clipped samples will have the common name 'partialWBC'.
            className1 = 'partialWBC'

            newSample, newMask = sample1, mask1

            # Whether the clipping will happen at the side or the corner of the
            # image, will be again selected randomly.
            number2 = np.random.randint( 100 )
                
            # The sample will be affixed in a location on the bg, such that 
            # it gets clipped by half. The clipping is always done by half 
            # because, what matters during this clipping is that, the sample 
            # should only be visible by a variable amount inside the image.
            # Now because of the variation of the size of the samples, they 
            # will anyway be visible by variable amount inside the image 
            # even if the percentage of clipping is kept constant. So to 
            # keep things simple the clipping is always done by 50%.
            # Because of the same reason as stated in the previous case, 
            # the clipping at the corners is kept constant at 25% only. The 
            # variability in size of the samples will take care of the rest.
            
            # Here the clipping is only done in the left margin and in the 
            # bottom left corner. After the sample2 is pasted, the final image
            # will be rotated randomly. That will result in having a clipping 
            # effect on all the margins and corners.

            newSampleH, newSampleW, _ = newSample.shape

            tlX = newBgW - int( newSampleW * 0.5 )
            
            if number2 < 60:
                # Clip at the left side of the image.
                tlY = np.random.randint( newBgH - newSampleH )

#-------------------------------------------------------------------------------

            else:    # Clip at the bottom right corner.
                tlY = newBgH - int( newSampleH * 0.5 )
                
            # Fixing the sample onto the background.
            image, posX1, posY1, bboxW1, bboxH1 = fixSampleToBg( newSample, newMask, newBg1, tlX, tlY )
            # Now remove the rbc cells which are overlapped by this wbc cell.
            newRbcOnCurrentBg = { k : v for k, v in rbcOnCurrentBg.items() \
                                  if abs(posX1-v['posX']) > bboxW1/2 or abs(posY1-v['posY']) > bboxH1/2 }
            rbcOnCurrentBg = newRbcOnCurrentBg
            
            # Create the segmented label image as well if createSegmentLabelImg is True:
            if createSegmentLabelImg:
                sampleColor = classNameToColor[ className1 ]
                sampleSegImg = cv2.bitwise_and( np.array( sampleColor ), newMask )
                segImg1, _, _, _, _ = fixSampleToBg( sampleSegImg, newMask, newBgSegImg1, tlX, tlY )
                
#-------------------------------------------------------------------------------
                
        # If the clipSample is False, then the other augmentations like flipping
        # and rotations are done.

        elif doHoriFlip or doVertFlip or do90degFlips or doRandomRot:

            # Augmenting the samples before affixing onto the background.
            
            # There are altogether 4 kinds of augmentation that this function can 
            # do, doRandomRot, doHoriFlip, doVertFlip, do90degFlips and no 
            # augmentation.
            # What kind of augmentation is to be done for this sample is chosen 
            # at random with a equal probability (20% for each type).
            # However, if the type of augmentation chosen doen not have it's 
            # corresponding flag True, then no augmentation is done.
    
            number = np.random.randint( 100 )
            
            # Horizontal flip sample1.
            
            if number < 20 and doHoriFlip:
#                print(sample1.shape, mask1.shape)##################
                
                newSample, newMask = horiFlipSampleAndMask( sample1, mask1 )
                bboxH1, bboxW1, _ = newSample.shape

#-------------------------------------------------------------------------------
    
            # Vertical flip sample1.
    
            elif number >= 20 and number < 40 and doVertFlip:
#                print(sample1.shape, mask1.shape)##################
                
                newSample, newMask = vertFlipSampleAndMask( sample1, mask1 )
                bboxH1, bboxW1, _ = newSample.shape

#-------------------------------------------------------------------------------
    
            # 90 deg flip sample1.
    
            elif number >= 40 and number < 60 and do90degFlips:
                # Now the selection of whether the flip should be by 90, 180 or 270
                # deg, is done randomly (with equal probablity).                
#                print(sample1.shape, mask1.shape)##################
                
                newSample, newMask, bboxW1, bboxH1 = random90degFlipSampleAndMask( sample1, mask1 )
                
#-------------------------------------------------------------------------------
    
            # Rotation by random angles sample1.
    
            elif number >= 60 and number < 80 and doRandomRot:
                # During rotation by arbitrary angles, the sample first needs to be
                # pasted on a bigger blank array, otherwise it will get cropped 
                # due to rotation.
#                print(sample1.shape, mask1.shape)##################
                
                newSample, newMask, bboxW1, bboxH1 = randomRotationSampleAndMask( sample1, mask1 )
                
#-------------------------------------------------------------------------------
                
            # No augmentation sample1.
            
            else:
                newSample, newMask = sample1, mask1
                bboxH1, bboxW1, _ = newSample.shape
            
            # x, y of top left corner of the region where sample will be pasted.
            newSampleH, newSampleW, _ = newSample.shape
            tlY = np.random.randint( newBgH - newSampleH )
            tlX = np.random.randint( max( newBgW * 0.5 - newSampleW, 1 ) ) \
                                                + int( newBgW * 0.5 ) 
            
            # Fixing the sample onto the background.
            image, posX1, posY1, _, _ = fixSampleToBg( newSample, newMask, newBg1, tlX, tlY )
            # Now remove the rbc cells which are overlapped by this wbc cell.
            newRbcOnCurrentBg = { k : v for k, v in rbcOnCurrentBg.items() \
                                  if abs(posX1-v['posX']) > bboxW1/2 or abs(posY1-v['posY']) > bboxH1/2 }
            rbcOnCurrentBg = newRbcOnCurrentBg

            # Create the segmented label image as well if createSegmentLabelImg is True:
            if createSegmentLabelImg:
                sampleColor = classNameToColor[ className1 ]
                sampleSegImg = cv2.bitwise_and( np.array( sampleColor ), newMask )
                segImg1, _, _, _, _ = fixSampleToBg( sampleSegImg, newMask, newBgSegImg1, tlX, tlY )
                        
#-------------------------------------------------------------------------------
        
        # If both the clipSample1 and the other augmentation flags are False, 
        # then no augmentation is performed.

        else:
            newSample, newMask = sample1, mask1
        
            # x, y of top left corner of the region where sample will be pasted.
            newSampleH, newSampleW, _ = newSample.shape
            tlY = np.random.randint( newBgH - newSampleH )
            tlX = np.random.randint( max( newBgW * 0.5 - newSampleW, 1 ) ) \
                                                + int( newBgW * 0.5 ) 
            # Fixing the sample onto the background.
            image, posX1, posY1, bboxW1, bboxH1 = fixSampleToBg( newSample, newMask, newBg1, tlX, tlY )
            # Now remove the rbc cells which are overlapped by this wbc cell.
            newRbcOnCurrentBg = { k : v for k, v in rbcOnCurrentBg.items() \
                                  if abs(posX1-v['posX']) > bboxW1/2 or abs(posY1-v['posY']) > bboxH1/2 }
            rbcOnCurrentBg = newRbcOnCurrentBg

            # Create the segmented label image as well if createSegmentLabelImg is True:
            if createSegmentLabelImg:
                sampleColor = classNameToColor[ className1 ]
                sampleSegImg = cv2.bitwise_and( np.array( sampleColor ), newMask )
                segImg1, _, _, _, _ = fixSampleToBg( sampleSegImg, newMask, newBgSegImg1, tlX, tlY )
                
#-------------------------------------------------------------------------------

        # The new background for sample2 will be the image formed earlier where
        # the sample1 was affixed onto the background.
        newBg2 = image
        newBgH, newBgW, _ = newBg2.shape
        
        if createSegmentLabelImg:   newBgSegImg2 = segImg1
        
        # Now, to have a variation in the position between sample1 and sample2,
        # the current image (which is the same as the newBg2) is rotated by 
        # multiples of 90 deg. And along with that the location where sample1 
        # has been pasted (given by the current values of tlX, tlY) has to be 
        # considered, so that the sample2 does not overlap with sample1.
        # The rotations will be done randomly, and the tlX and tlY values will
        # thereby, change to different values in each case. All these should be
        # taken into account.
        
        # Now the selection of whether the flip should be by 90, 180 or 270
        # deg, is done randomly (with equal probablity).
        number1 = np.random.randint( 100 )

        if number1 < 33:
            # Flip by 90 deg (same as horizontal flip + transpose).
            newBg2 = cv2.transpose( cv2.flip( newBg2, 1 ) )
            posX1, posY1 = posY1, imgW - posX1
            bboxW1, bboxH1 = bboxH1, bboxW1
            hLimit, hOffset = tlX, newBgW - tlX
            wLimit, wOffset = newBgW, 0
            # Now modifying the rbc cell locations.
            for k, v in rbcOnCurrentBg.items():
                v['posX'], v['posY'] = v['posY'], imgW - v['posX']
                v['bboxW'], v['bboxH'] = v['bboxH'], v['bboxW']
                
            if createSegmentLabelImg:
                newBgSegImg2 = cv2.transpose( cv2.flip( newBgSegImg2, 1 ) )

            
        elif number1 >= 33 and number1 < 66:
            # Flip by 180 deg (same as horizontal flip + vertical flip).
            newBg2 = cv2.flip( newBg2, -1 )
            posX1, posY1 = imgW - posX1, imgH - posY1
            hLimit, hOffset = newBgH, 0
            wLimit, wOffset = tlX, newBgW - tlX
            # Now modifying the rbc cell locations.
            for k, v in rbcOnCurrentBg.items():
                v['posX'], v['posY'] = imgW - v['posX'], imgH - v['posY']

            if createSegmentLabelImg:
                newBgSegImg2 = cv2.flip( newBgSegImg2, -1 )

                        
        else:   # Flip by 270 deg (same as vertical flip + transpose).
            newBg2 = cv2.transpose( cv2.flip( newBg2, 0 ) )
            posX1, posY1 = imgH - posY1, posX1
            bboxW1, bboxH1 = bboxH1, bboxW1
            hLimit, hOffset = tlX, 0
            wLimit, wOffset = newBgW, 0
            # Now modifying the rbc cell locations.
            for k, v in rbcOnCurrentBg.items():
                v['posX'], v['posY'] = imgH - v['posY'], v['posX']
                v['bboxW'], v['bboxH'] = v['bboxH'], v['bboxW']
                
            if createSegmentLabelImg:
                newBgSegImg2 = cv2.transpose( cv2.flip( newBgSegImg2, 0 ) )

#-------------------------------------------------------------------------------

        # Clip sample2 at the image boundary.

        # If the clipSample flag is True, then the other data augmentation like
        # flipping and rotation is ignored (even if their corresponding flags
        # are True). As this does not make much of a difference.

        if clipSample2:
            # All clipped samples will have the common name 'partialWBC'.
            className2 = 'partialWBC'

            newSample, newMask = sample2, mask2

            # Whether the clipping will happen at the side or the corner of the
            # image, will be again selected randomly.
            number2 = np.random.randint( 100 )
                
            # The sample will be affixed in a location on the bg, such that 
            # it gets clipped by half. The clipping is always done by half 
            # because, what matters during this clipping is that, the sample 
            # should only be visible by a variable amount inside the image.
            # Now because of the variation of the size of the samples, they 
            # will anyway be visible by variable amount inside the image 
            # even if the percentage of clipping is kept constant. So to 
            # keep things simple the clipping is always done by 50%.
            # Because of the same reason as stated in the previous case, 
            # the clipping at the corners is kept constant at 25% only. The 
            # variability in size of the samples will take care of the rest.
            
            newSampleH, newSampleW, _ = newSample.shape

            tlX = newBgW - int( newSampleW * 0.5 )

            if number2 < 15:
                # Clip at the left side of the image.
                tlY = np.random.randint( max( hLimit - newSampleH, 1 ) ) + hOffset
                # Fixing the sample onto the background.
                image, posX2, posY2, bboxW2, bboxH2 = fixSampleToBg( newSample, newMask, newBg2, tlX, tlY )
                # Now remove the rbc cells which are overlapped by this wbc cell.
                newRbcOnCurrentBg = { k : v for k, v in rbcOnCurrentBg.items() \
                                      if abs(posX2-v['posX']) > bboxW2/2 or abs(posY2-v['posY']) > bboxH2/2 }
                rbcOnCurrentBg = newRbcOnCurrentBg

                # Create the segmented label image as well if createSegmentLabelImg is True:
                if createSegmentLabelImg:
                    sampleColor = classNameToColor[ className2 ]
                    sampleSegImg = cv2.bitwise_and( np.array( sampleColor ), newMask )
                    segImg, _, _, _, _ = fixSampleToBg( sampleSegImg, newMask, newBgSegImg2, tlX, tlY )
                

            elif number2 >= 15 and number2 < 30:
                # Clip at the top side of the image (which is same as clip on 
                # left + flip by 90 deg).
                tlY = np.random.randint( max( hLimit - newSampleH, 1 ) ) + hOffset
                # Fixing the sample onto the background.
                image, posX2, posY2, bboxW2, bboxH2 = fixSampleToBg( newSample, newMask, newBg2, tlX, tlY )
                # Now remove the rbc cells which are overlapped by this wbc cell.
                newRbcOnCurrentBg = { k : v for k, v in rbcOnCurrentBg.items() \
                                      if abs(posX2-v['posX']) > bboxW2/2 or abs(posY2-v['posY']) > bboxH2/2 }
                rbcOnCurrentBg = newRbcOnCurrentBg

                image = cv2.transpose( cv2.flip( image, 1 ) )
                posX2, posY2 = posY2, imgW - posX2
                bboxW2, bboxH2 = bboxH2, bboxW2
                posX1, posY1 = posY1, imgW - posX1
                bboxW1, bboxH1 = bboxH1, bboxW1
                # Now modifying the rbc cell locations.
                for k, v in rbcOnCurrentBg.items():
                    v['posX'], v['posY'] = v['posY'], imgW - v['posX']
                    v['bboxW'], v['bboxH'] = v['bboxH'], v['bboxW']

                # Create the segmented label image as well if createSegmentLabelImg is True:
                if createSegmentLabelImg:
                    sampleColor = classNameToColor[ className2 ]
                    sampleSegImg = cv2.bitwise_and( np.array( sampleColor ), newMask )
                    segImg, _, _, _, _ = fixSampleToBg( sampleSegImg, newMask, newBgSegImg2, tlX, tlY )
                    segImg = cv2.transpose( cv2.flip( segImg, 1 ) )
                
            
            elif number2 >= 30 and number2 < 45:
                # Clip at the right side of the image (which is same as clip on 
                # left + flip by 180 deg).
                tlY = np.random.randint( max( hLimit - newSampleH, 1 ) ) + hOffset
                # Fixing the sample onto the background.
                image, posX2, posY2, bboxW2, bboxH2 = fixSampleToBg( newSample, newMask, newBg2, tlX, tlY )
                # Now remove the rbc cells which are overlapped by this wbc cell.
                newRbcOnCurrentBg = { k : v for k, v in rbcOnCurrentBg.items() \
                                      if abs(posX2-v['posX']) > bboxW2/2 or abs(posY2-v['posY']) > bboxH2/2 }
                rbcOnCurrentBg = newRbcOnCurrentBg

                image = cv2.flip( image, -1 )
                posX2, posY2 = imgW - posX2, imgH - posY2
                posX1, posY1 = imgW - posX1, imgH - posY1
                # Now modifying the rbc cell locations.
                for k, v in rbcOnCurrentBg.items():
                    v['posX'], v['posY'] = imgW - v['posX'], imgH - v['posY']
            
                # Create the segmented label image as well if createSegmentLabelImg is True:
                if createSegmentLabelImg:
                    sampleColor = classNameToColor[ className2 ]
                    sampleSegImg = cv2.bitwise_and( np.array( sampleColor ), newMask )
                    segImg, _, _, _, _ = fixSampleToBg( sampleSegImg, newMask, newBgSegImg2, tlX, tlY )
                    segImg = cv2.flip( segImg, -1 )
                    
            
            elif number2 >= 45 and number2 < 60:
                # Clip at the bottom side of the image (which is same as clip on 
                # left + flip by 270 deg).
                tlY = np.random.randint( max( hLimit - newSampleH, 1 ) ) + hOffset
                # Fixing the sample onto the background.
                image, posX2, posY2, bboxW2, bboxH2 = fixSampleToBg( newSample, newMask, newBg2, tlX, tlY )
                # Now remove the rbc cells which are overlapped by this wbc cell.
                newRbcOnCurrentBg = { k : v for k, v in rbcOnCurrentBg.items() \
                                      if abs(posX2-v['posX']) > bboxW2/2 or abs(posY2-v['posY']) > bboxH2/2 }
                rbcOnCurrentBg = newRbcOnCurrentBg

                image = cv2.transpose( cv2.flip( image, 0 ) )
                posX2, posY2 = imgH - posY2, posX2
                bboxW2, bboxH2 = bboxH2, bboxW2
                posX1, posY1 = imgH - posY1, posX1
                bboxW1, bboxH1 = bboxH1, bboxW1
                # Now modifying the rbc cell locations.
                for k, v in rbcOnCurrentBg.items():
                    v['posX'], v['posY'] = imgH - v['posY'], v['posX']
                    v['bboxW'], v['bboxH'] = v['bboxH'], v['bboxW']

                # Create the segmented label image as well if createSegmentLabelImg is True:
                if createSegmentLabelImg:
                    sampleColor = classNameToColor[ className2 ]
                    sampleSegImg = cv2.bitwise_and( np.array( sampleColor ), newMask )
                    segImg, _, _, _, _ = fixSampleToBg( sampleSegImg, newMask, newBgSegImg2, tlX, tlY )
                    segImg = cv2.transpose( cv2.flip( segImg, 0 ) )
                
#-------------------------------------------------------------------------------

            elif number2 >= 60 and number2 < 70:
                # Clip at the bottom right corner.
                tlY = hLimit - int( newSampleH * 0.5 ) + hOffset
                # Fixing the sample onto the background.
                image, posX2, posY2, bboxW2, bboxH2 = fixSampleToBg( newSample, newMask, newBg2, tlX, tlY )
                # Now remove the rbc cells which are overlapped by this wbc cell.
                newRbcOnCurrentBg = { k : v for k, v in rbcOnCurrentBg.items() \
                                      if abs(posX2-v['posX']) > bboxW2/2 or abs(posY2-v['posY']) > bboxH2/2 }
                rbcOnCurrentBg = newRbcOnCurrentBg
                
                # Create the segmented label image as well if createSegmentLabelImg is True:
                if createSegmentLabelImg:
                    sampleColor = classNameToColor[ className2 ]
                    sampleSegImg = cv2.bitwise_and( np.array( sampleColor ), newMask )
                    segImg, _, _, _, _ = fixSampleToBg( sampleSegImg, newMask, newBgSegImg2, tlX, tlY )
                    

            elif number2 >= 70 and number2 < 80:
                # Clip at the top right corner (which is same as clip on the 
                # bottom right corner + flip by 90 deg). 
                tlY = hLimit - int( newSampleH * 0.5 ) + hOffset
                # Fixing the sample onto the background.
                image, posX2, posY2, bboxW2, bboxH2 = fixSampleToBg( newSample, newMask, newBg2, tlX, tlY )
                # Now remove the rbc cells which are overlapped by this wbc cell.
                newRbcOnCurrentBg = { k : v for k, v in rbcOnCurrentBg.items() \
                                      if abs(posX2-v['posX']) > bboxW2/2 or abs(posY2-v['posY']) > bboxH2/2 }
                rbcOnCurrentBg = newRbcOnCurrentBg

                image = cv2.transpose( cv2.flip( image, 1 ) )
                posX2, posY2 = posY2, imgW - posX2
                bboxW2, bboxH2 = bboxH2, bboxW2            
                posX1, posY1 = posY1, imgW - posX1
                bboxW1, bboxH1 = bboxH1, bboxW1
                # Now modifying the rbc cell locations.
                for k, v in rbcOnCurrentBg.items():
                    v['posX'], v['posY'] = v['posY'], imgW - v['posX']
                    v['bboxW'], v['bboxH'] = v['bboxH'], v['bboxW']
                
                # Create the segmented label image as well if createSegmentLabelImg is True:
                if createSegmentLabelImg:
                    sampleColor = classNameToColor[ className2 ]
                    sampleSegImg = cv2.bitwise_and( np.array( sampleColor ), newMask )
                    segImg, _, _, _, _ = fixSampleToBg( sampleSegImg, newMask, newBgSegImg2, tlX, tlY )
                    segImg = cv2.transpose( cv2.flip( segImg, 1 ) )
                            

            elif number2 >= 80 and number2 < 90:
                # Clip at the top left corner (which is same as clip on the 
                # bottom right corner + flip by 180 deg). 
                tlY = hLimit - int( newSampleH * 0.5 ) + hOffset
                # Fixing the sample onto the background.
                image, posX2, posY2, bboxW2, bboxH2 = fixSampleToBg( newSample, newMask, newBg2, tlX, tlY )
                # Now remove the rbc cells which are overlapped by this wbc cell.
                newRbcOnCurrentBg = { k : v for k, v in rbcOnCurrentBg.items() \
                                      if abs(posX2-v['posX']) > bboxW2/2 or abs(posY2-v['posY']) > bboxH2/2 }
                rbcOnCurrentBg = newRbcOnCurrentBg

                image = cv2.flip( image, -1 )
                posX2, posY2 = imgW - posX2, imgH - posY2
                posX1, posY1 = imgW - posX1, imgH - posY1
                # Now modifying the rbc cell locations.
                for k, v in rbcOnCurrentBg.items():
                    v['posX'], v['posY'] = imgW - v['posX'], imgH - v['posY']

                # Create the segmented label image as well if createSegmentLabelImg is True:
                if createSegmentLabelImg:
                    sampleColor = classNameToColor[ className2 ]
                    sampleSegImg = cv2.bitwise_and( np.array( sampleColor ), newMask )
                    segImg, _, _, _, _ = fixSampleToBg( sampleSegImg, newMask, newBgSegImg2, tlX, tlY )
                    segImg = cv2.flip( segImg, -1 )
                            

            elif number2 >= 90 and number2 < 100:
                # Clip at the bottom left corner (which is same as clip on the 
                # bottom right corner + flip by 270 deg). 
                tlY = hLimit - int( newSampleH * 0.5 ) + hOffset
                # Fixing the sample onto the background.
                image, posX2, posY2, bboxW2, bboxH2 = fixSampleToBg( newSample, newMask, newBg2, tlX, tlY )
                # Now remove the rbc cells which are overlapped by this wbc cell.
                newRbcOnCurrentBg = { k : v for k, v in rbcOnCurrentBg.items() \
                                      if abs(posX2-v['posX']) > bboxW2/2 or abs(posY2-v['posY']) > bboxH2/2 }
                rbcOnCurrentBg = newRbcOnCurrentBg

                image = cv2.transpose( cv2.flip( image, 0 ) )
                posX2, posY2 = imgH - posY2, posX2
                bboxW2, bboxH2 = bboxH2, bboxW2
                posX1, posY1 = imgH - posY1, posX1
                bboxW1, bboxH1 = bboxH1, bboxW1
                # Now modifying the rbc cell locations.
                for k, v in rbcOnCurrentBg.items():
                    v['posX'], v['posY'] = imgH - v['posY'], v['posX']
                    v['bboxW'], v['bboxH'] = v['bboxH'], v['bboxW']

                # Create the segmented label image as well if createSegmentLabelImg is True:
                if createSegmentLabelImg:
                    sampleColor = classNameToColor[ className2 ]
                    sampleSegImg = cv2.bitwise_and( np.array( sampleColor ), newMask )
                    segImg, _, _, _, _ = fixSampleToBg( sampleSegImg, newMask, newBgSegImg2, tlX, tlY )
                    segImg = cv2.transpose( cv2.flip( segImg, 0 ) )

#-------------------------------------------------------------------------------
                
        # If the clipSample is False, then the other augmentations like flipping
        # and rotations are done.

        elif doHoriFlip or doVertFlip or do90degFlips or doRandomRot:
            
            # Augmenting the samples before affixing onto the background.
            
            # There are altogether 4 kinds of augmentation that this function can 
            # do, doRandomRot, doHoriFlip, doVertFlip, do90degFlips and no 
            # augmentation.
            # What kind of augmentation is to be done for this sample is chosen 
            # at random with a equal probability (20% for each type).
            # However, if the type of augmentation chosen doen not have it's 
            # corresponding flag True, then no augmentation is done.

            number = np.random.randint( 100 )
            
            # Horizontal flip sample2.
            
            if number < 20 and doHoriFlip:
#                print(2, sample2.shape, mask2.shape)##################
                
                newSample, newMask = horiFlipSampleAndMask( sample2, mask2 )
                bboxH2, bboxW2, _ = newSample.shape

#-------------------------------------------------------------------------------
    
            # Vertical flip sample2.
    
            elif number >= 20 and number < 40 and doVertFlip:
#                print(2, sample2.shape, mask2.shape)##################

                newSample, newMask = vertFlipSampleAndMask( sample2, mask2 )
                bboxH2, bboxW2, _ = newSample.shape

#-------------------------------------------------------------------------------
    
            # 90 deg flip sample2.
    
            elif number >= 40 and number < 60 and do90degFlips:
                # Now the selection of whether the flip should be by 90, 180 or 270
                # deg, is done randomly (with equal probablity).                
#                print(2, sample2.shape, mask2.shape)##################

                newSample, newMask, bboxW2, bboxH2 = random90degFlipSampleAndMask( sample2, mask2 )
                
#-------------------------------------------------------------------------------
    
            # Rotation by random angles sample2.
    
            elif number >= 60 and number < 80 and doRandomRot:
                # During rotation by arbitrary angles, the sample first needs to be
                # pasted on a bigger blank array, otherwise it will get cropped 
                # due to rotation.
#                print(2, sample2.shape, mask2.shape)##################

                newSample, newMask, bboxW2, bboxH2 = randomRotationSampleAndMask( sample2, mask2 )
           
#-------------------------------------------------------------------------------
                
            # No augmentation sample2.
            
            else:
                newSample, newMask = sample2, mask2
                bboxH2, bboxW2, _ = newSample.shape
            
            # x, y of top left corner of the region where sample will be pasted.
            newSampleH, newSampleW, _ = newSample.shape
            tlY = np.random.randint( max( hLimit - newSampleH, 1 ) ) + hOffset
            tlX = np.random.randint( max( wLimit - newSampleW, 1 ) ) + wOffset 
                
            # Fixing the sample onto the background.
            image, posX2, posY2, _, _ = fixSampleToBg( newSample, newMask, newBg2, tlX, tlY )
            # Now remove the rbc cells which are overlapped by this wbc cell.
            newRbcOnCurrentBg = { k : v for k, v in rbcOnCurrentBg.items() \
                                  if abs(posX2-v['posX']) > bboxW2/2 or abs(posY2-v['posY']) > bboxH2/2 }
            rbcOnCurrentBg = newRbcOnCurrentBg
            
            # Create the segmented label image as well if createSegmentLabelImg is True:
            if createSegmentLabelImg:
                sampleColor = classNameToColor[ className2 ]
                sampleSegImg = cv2.bitwise_and( np.array( sampleColor ), newMask )
                segImg, _, _, _, _ = fixSampleToBg( sampleSegImg, newMask, newBgSegImg2, tlX, tlY )
            
#-------------------------------------------------------------------------------
        
        # If both the clipSample2 and the other augmentation flags are False, 
        # then no augmentation is performed.

        else:
            newSample, newMask = sample2, mask2
        
            # x, y of top left corner of the region where sample will be pasted.
            newSampleH, newSampleW, _ = newSample.shape
            tlY = np.random.randint( max( hLimit - newSampleH, 1 ) ) + hOffset
            tlX = np.random.randint( max( wLimit - newSampleW, 1 ) ) + wOffset 
            # Fixing the sample onto the background.
            image, posX2, posY2, bboxW2, bboxH2 = fixSampleToBg( newSample, newMask, newBg2, tlX, tlY )
            # Now remove the rbc cells which are overlapped by this wbc cell.
            newRbcOnCurrentBg = { k : v for k, v in rbcOnCurrentBg.items() \
                                  if abs(posX2-v['posX']) > bboxW2/2 or abs(posY2-v['posY']) > bboxH2/2 }
            rbcOnCurrentBg = newRbcOnCurrentBg
            
            # Create the segmented label image as well if createSegmentLabelImg is True:
            if createSegmentLabelImg:
                sampleColor = classNameToColor[ className2 ]
                sampleSegImg = cv2.bitwise_and( np.array( sampleColor ), newMask )
                segImg, _, _, _, _ = fixSampleToBg( sampleSegImg, newMask, newBgSegImg2, tlX, tlY )
                
##-------------------------------------------------------------------------------
#
#        cv2.imshow( 'sample', sample )
#        cv2.imshow( 'newSample', newSample )
#        cv2.imshow( 'mask', mask )
#        cv2.imshow( 'newMask', newMask )
#        cv2.imshow( 'newBg', newBg )
#        cv2.waitKey(0)
#                     
#-------------------------------------------------------------------------------

        # Saving the image.
        idx = nAlreadyExistingFiles + i    # This is the image index.
    
        imgSaveName = className1[:4] + '_' + \
                      className2[:4] + '_' + \
                      saveNameSuffix + '_' + str( idx ) + '.bmp'
                      
        cv2.imwrite( os.path.join( imgSaveLoc, imgSaveName ), image )

        # Saving the segmented image label as well if createSegmentLabelImg is True.
        if createSegmentLabelImg:
            segImgSaveName = className1[:4] + '_' + \
                          className2[:4] + '_seg' + '_' + \
                          saveNameSuffix + '_' + str( idx ) + '.bmp'
            cv2.imwrite( os.path.join( segmentSaveLoc, segImgSaveName ), segImg )
        
        # Creating the label json file.
        labelSaveName = className1[:4] + '_' + \
                        className2[:4] + '_' + \
                        saveNameSuffix + '_' + str( idx ) + '.json'
        
        classIdx1 = classNameToIdx[ className1 ]
        classIdx2 = classNameToIdx[ className2 ]
        
        infoDict = {}
        
        with open( os.path.join( labelSaveLoc, labelSaveName ), 'w' ) as infoFile:
            
#-------------------------------------------------------------------------------

            posX, posY, bboxW, bboxH = posX1, posY1, bboxW1, bboxH1
            className, classIdx = className1, classIdx1
            sampleLoc, sampleName = sampleLoc1, sampleName1

#-------------------------------------------------------------------------------

            # Make sure the coordinates are inside the boundaries of the image.
            if posX >= imgW:      posX = imgW - 1
            if posX < 0:            posX = 0
            if posY >= imgH:      posY = imgH - 1
            if posY < 0:            posY = 0
            tlX, tlY = posX-bboxW*0.5, posY-bboxH*0.5   # Top left corner.
            brX, brY = posX+bboxW*0.5, posY+bboxH*0.5   # Bottom right corner.
            if tlX < 0:            tlX = 0
            if tlY < 0:            tlY = 0
            if brX >= imgW:      brX = imgW - 1
            if brY >= imgH:      brY = imgH - 1
            bboxW, bboxH = int( brX - tlX ), int( brY - tlY )   # Update box size.
            
#-------------------------------------------------------------------------------

            infoDict[0] = {
                            'className': className, 'classIdx': classIdx, \
                            'posX': int(posX), 'posY': int(posY), \
                            'bboxW': bboxW, 'bboxH': bboxH, \
                            'tlX': int(tlX), 'tlY': int(tlY), \
                            'samplePath': os.path.join( sampleLoc, sampleName ), \
                            'bgPath': os.path.join( bgLoc, bgName ) \
                          }
            
#-------------------------------------------------------------------------------

            posX, posY, bboxW, bboxH = posX2, posY2, bboxW2, bboxH2
            className, classIdx = className2, classIdx2
            sampleLoc, sampleName = sampleLoc2, sampleName2

#-------------------------------------------------------------------------------

            # Make sure the coordinates are inside the boundaries of the image.
            if posX >= imgW:      posX = imgW - 1
            if posX < 0:            posX = 0
            if posY >= imgH:      posY = imgH - 1
            if posY < 0:            posY = 0
            tlX, tlY = posX-bboxW*0.5, posY-bboxH*0.5   # Top left corner.
            brX, brY = posX+bboxW*0.5, posY+bboxH*0.5   # Bottom right corner.
            if tlX < 0:            tlX = 0
            if tlY < 0:            tlY = 0
            if brX >= imgW:      brX = imgW - 1
            if brY >= imgH:      brY = imgH - 1
            bboxW, bboxH = int( brX - tlX ), int( brY - tlY )   # Update box size.
            
#-------------------------------------------------------------------------------

            infoDict[1] = {
                            'className': className, 'classIdx': classIdx, \
                            'posX': int(posX), 'posY': int(posY), \
                            'bboxW': bboxW, 'bboxH': bboxH, \
                            'tlX': int(tlX), 'tlY': int(tlY), \
                            'samplePath': os.path.join( sampleLoc, sampleName ), \
                            'bgPath': os.path.join( bgLoc, bgName ) \
                          }
            
#-------------------------------------------------------------------------------

            # Now recording the rbc cells into the infoDict.
            nWbc = len( infoDict )      # Number of wbc cell records in infoDict.
#            print( len(rbcOnCurrentBg) )
            
            for r, (k, v) in enumerate( rbcOnCurrentBg.items() ):
                # Creating key for the rbc cell records. 
                # This makes sure that they are different from the keys of the 
                # wbc cell records, else they may overlap the wbc record.
                index = r + nWbc
                
#-------------------------------------------------------------------------------

                posX, posY, bboxW, bboxH = v['posX'], v['posY'], v['bboxW'], v['bboxH']
                classNameRbc, classIdxRbc = v['className'], v['classIdx']
                
#-------------------------------------------------------------------------------

                # Make sure the coordinates are inside the boundaries of the image.
                if posX >= imgW:      posX = imgW - 1
                if posX < 0:            posX = 0
                if posY >= imgH:      posY = imgH - 1
                if posY < 0:            posY = 0
                tlX, tlY = posX-bboxW*0.5, posY-bboxH*0.5   # Top left corner.
                brX, brY = posX+bboxW*0.5, posY+bboxH*0.5   # Bottom right corner.
                if tlX < 0:            tlX = 0
                if tlY < 0:            tlY = 0
                if brX >= imgW:      brX = imgW - 1
                if brY >= imgH:      brY = imgH - 1
                bboxW, bboxH = int( brX - tlX ), int( brY - tlY )   # Update box size.
            
#-------------------------------------------------------------------------------

                infoDict[ index ] = {
                                        'className': classNameRbc, 'classIdx': classIdxRbc, \
                                        'posX': int(posX), 'posY': int(posY), \
                                        'bboxW': bboxW, 'bboxH': bboxH, \
                                        'tlX': int(tlX), 'tlY': int(tlY), \
                                        'samplePath': None, \
                                        'bgPath': os.path.join( bgLoc, bgName ) \
                                    }
                
            json.dump( infoDict, infoFile, indent=4, separators=(',', ': ') )

#-------------------------------------------------------------------------------

        for k, v in infoDict.items():
            cv2.circle( image, (v['posX'], v['posY']), 2, (0,255,0), 2 )
            if v['className'] != '_':
                cv2.rectangle( image, (v['tlX'], v['tlY']), (v['tlX']+v['bboxW'], \
                                       v['tlY']+v['bboxH']), (0,255,0), 2 )
        cv2.imshow( 'image', image )
        # Show the segment label as well if the createSegmentLabelImg is True.
        if createSegmentLabelImg:   cv2.imshow( 'segment label', segImg )
        cv2.waitKey(30)

#===============================================================================

def blankBackground( bgLoc=None, imgSaveLoc=None, labelSaveLoc=None, \
                     nImgs=None, imgH=None, imgW=None,
                     saveNameSuffix=None, includeRbc=False, createSegmentLabelImg=False, \
                     segmentSaveLoc=None ):
    '''
    This function creates images where there are no wbc cells. Only a background
    image of rbc cells.
    These images are created by taking backgrounds from bgLoc.
    The backgrounds are selected randomly from the available collection. 
    Total number of images created is nImgs. Images are saved in the imgSaveLoc.
    The labels of the corresponding images are also created as json files in 
    the labelSaveLoc.
    imgH and imgW defines the size of the image to be created.
    The saveNameSuffix is a string, that will be appended to the name of the 
    image file while saving. This is important to identify the dataset from 
    where the image has been synthesized.
    The function also checks if there are already some rbc annotations present 
    on the background or not. If so then it includes them in the labels.
    The createSegmentLabelImg indicates if a semantic segmentation label image 
    has to be created as well. The colors of the segments for different objects 
    are mentioned in the global variables. Segment save location is also provided.
    '''
    
    if bgLoc is None or imgSaveLoc is None or labelSaveLoc is None \
       or nImgs is None or imgH is None or imgW is None or saveNameSuffix is None:
           print( '\nERROR: one or more input arguments missing ' \
                  'in blankBackground. Aborting.\n' )
           sys.exit()
    
    if createSegmentLabelImg:
        if segmentSaveLoc is None:
            print( '\nERROR: one or more input arguments missing ' \
                   'in blankBackground for segments. Aborting.\n' )
            sys.exit()
    
#-------------------------------------------------------------------------------
        
    # Checking if there is any label file for the bg images present or not. 
    # These files will include rbc annotations, if present.
    imgFolderParentDir = '\\'.join( bgLoc.split('\\')[:-1] )
    imgFolderName = bgLoc.split('\\')[-1]
    labelFolderName = imgFolderName + '_labels'
    labelFolderLoc = os.path.join( imgFolderParentDir, labelFolderName )
    
    if createSegmentLabelImg:
        bgSegmentFolderName = imgFolderName + '_segments'
        bgSegmentFolderLoc = os.path.join( imgFolderParentDir, bgSegmentFolderName )

#-------------------------------------------------------------------------------
    
    # Number of files already existing in the imgSaveLoc is calculated. This 
    # will be used to assign the index to the file while saving.
    nAlreadyExistingFiles = len( os.listdir( imgSaveLoc ) )
    
    bgList = []
    
    # Creating the images.    
    for i in range( nImgs ):
        # Fill the lists if they are empty.
        # As a sample and a bg is used for creating an image, they are deleted
        # from this list. So if this list gets empty, then it is reinitialized.
        if len( bgList ) == 0:      bgList = os.listdir( bgLoc )
        
        # Select a background at random.
        bgIdx = np.random.randint( len( bgList ) )
        bgName = bgList[ bgIdx ]
        bg = cv2.imread( os.path.join( bgLoc, bgName ) )
        
        # Remove the entry of this bg from the respective lists so 
        # that they are not used again. It will only be used again if all the 
        # existing bg images in the lists are used up and the lists become empty.
        bgList.pop( bgIdx )

#-------------------------------------------------------------------------------
        
        # Setting the background.
        
        # It may happen that the bg image is larger than size imgH x imgW.
        # In that case, a imgH x imgW region is cropped out from the bg image.
        bgH, bgW, _ = bg.shape
        
        # Determining the x and y of the top left corner of the region to be
        # cropped out from the bg image.
        bgTlY = np.random.randint( bgH - imgH ) if bgH > imgH else 0
        bgTlX = np.random.randint( bgW - imgW ) if bgW > imgW else 0
        
        # IMPORTANT: The bg image must be larger or equal in size to imgH x imgW.
        newBg = bg[ bgTlY : bgTlY + imgH, bgTlX : bgTlX + imgW ]
        
        newBgH, newBgW, _ = newBg.shape

#-------------------------------------------------------------------------------

        # Also doing the same processing for the segmented image label.
        if createSegmentLabelImg:
            bgSegName = 'seg_' + '_'.join( bgName.split('_')[1:] )
            bgSegImg = cv2.imread( os.path.join( bgSegmentFolderLoc, bgSegName ) )
            
            newBgSegImg = bgSegImg[ bgTlY : bgTlY + imgH, bgTlX : bgTlX + imgW ]
            
#-------------------------------------------------------------------------------

        # Including the rbc annotations in the label if there are label files 
        # present for these background images.

        # If however the argument includeRbc is false, this function will keep 
        # this dictionary empty.
        if includeRbc:
            # Now those rbc which falls within the region which is cropped for 
            # bg image, are included inside another dictionary.
            # If there are no rbc annotations, then this dictionary will stay empty.
            rbcOnCurrentBg = {}

            if os.path.exists( labelFolderLoc ):
                bgLabelName = bgName[:-4] + '.json'
                bgLabelLoc = os.path.join( labelFolderLoc, bgLabelName )
                
                with open( bgLabelLoc, 'r' ) as infoFile:
                    infoDict = json.load( infoFile )
                
                for k, v in infoDict.items():
                    posX, posY = v['posX'], v['posY']
                    if posX >= bgTlX and posX < bgTlX + imgW and \
                       posY >= bgTlY and posY < bgTlY + imgH:
                           rbcOnCurrentBg[k] = v
                           rbcOnCurrentBg[k]['posX'] -= bgTlX
                           rbcOnCurrentBg[k]['posY'] -= bgTlY
                           rbcOnCurrentBg[k]['tlX'] -= bgTlX
                           rbcOnCurrentBg[k]['tlY'] -= bgTlY
                           
        else:
            rbcOnCurrentBg = {}

#-------------------------------------------------------------------------------

        # Saving the image.
        idx = nAlreadyExistingFiles + i    # This is the image index.
        
        imgSaveName = 'back' + '_' + \
                      saveNameSuffix + '_' + str( idx ) + '.bmp'
        # The file extension is explicitly specified here because, many of the 
        # background images have .jpg or .png file extension as well. So to 
        # make the saved file consistent, this is specified.
                
        cv2.imwrite( os.path.join( imgSaveLoc, imgSaveName ), newBg )
        
        # Saving the segmented image label as well if createSegmentLabelImg is True.
        if createSegmentLabelImg:
            segImgSaveName = 'back' + '_seg' + '_' + \
                             saveNameSuffix + '_' + str( idx ) + '.bmp'
            cv2.imwrite( os.path.join( segmentSaveLoc, segImgSaveName ), newBgSegImg )
                
        # Creating the label json file.
        labelSaveName = 'back' + '_' + \
                        saveNameSuffix + '_' + str( idx ) + '.json'
        
        infoDict = {}
        
        with open( os.path.join( labelSaveLoc, labelSaveName ), 'w' ) as infoFile:
            
            for index, (k, v) in enumerate( rbcOnCurrentBg.items() ):
                
#-------------------------------------------------------------------------------

                posX, posY, bboxW, bboxH = v['posX'], v['posY'], v['bboxW'], v['bboxH']
                classNameRbc, classIdxRbc = v['className'], v['classIdx']

#-------------------------------------------------------------------------------

                # Make sure the coordinates are inside the boundaries of the image.
                if posX >= imgW:      posX = imgW - 1
                if posX < 0:            posX = 0
                if posY >= imgH:      posY = imgH - 1
                if posY < 0:            posY = 0
                tlX, tlY = posX-bboxW*0.5, posY-bboxH*0.5   # Top left corner.
                brX, brY = posX+bboxW*0.5, posY+bboxH*0.5   # Bottom right corner.
                if tlX < 0:            tlX = 0
                if tlY < 0:            tlY = 0
                if brX >= imgW:      brX = imgW - 1
                if brY >= imgH:      brY = imgH - 1
                bboxW, bboxH = int( brX - tlX ), int( brY - tlY )   # Update box size.
            
#-------------------------------------------------------------------------------

                infoDict[ index ] = {
                                        'className': classNameRbc, 'classIdx': classIdxRbc, \
                                        'posX': int(posX), 'posY': int(posY), \
                                        'bboxW': bboxW, 'bboxH': bboxH, \
                                        'tlX': int(tlX), 'tlY': int(tlY), \
                                        'samplePath': None, \
                                        'bgPath': os.path.join( bgLoc, bgName ) \
                                    }
                
            json.dump( infoDict, infoFile, indent=4, separators=(',', ': ') )

#-------------------------------------------------------------------------------

        for k, v in infoDict.items():
            cv2.circle( newBg, ( v['posX'], v['posY'] ), 3, (0,255,0), -1 )
#        print(len(rbcOnCurrentBg))
        cv2.imshow( 'image', newBg )
        # Show the segment label as well if the createSegmentLabelImg is True.
        if createSegmentLabelImg:   cv2.imshow( 'segment label', newBgSegImg )
        cv2.waitKey(30)

#===============================================================================

def markPoints( event, x, y, flags, params ):
    '''
    This is a function that is called on mouse callback.
    '''
    global cBix, cBiy
    if event == cv2.EVENT_LBUTTONDOWN:
        cBix, cBiy = x, y

#===============================================================================

def selectPts( filePath=None ):
    '''
    This function opens the image and lets user select the points in it.
    These points are returned as a list.
    If the image is bigger than 800 x 600, it is displayed as 800 x 600. But
    the points are mapped and stored as per the original dimension of the image.
    The points are clicked by mouse on the image itself and they are stored in
    the listOfPts.
    '''
    
    global cBix, cBiy
    
    img = cv2.imread( filePath )
    h, w = img.shape[0], img.shape[1]
    
    w1, h1, wRatio, hRatio, resized = w, h, 1, 1, False
#    print( 'Image size: {}x{}'.format(w, h) )
    
    if w > 800:
        w1, resized = 800, True
        wRatio = w / w1
    if h > 600:
        h1, resized = 600, True
        hRatio = h / h1

    if resized:     img = cv2.resize( img, (w1, h1), interpolation=cv2.INTER_AREA )

    cv2.namedWindow( 'Image' )
    cv2.setMouseCallback( 'Image', markPoints )  # Function to detect mouseclick
    key = ord('`')

#-------------------------------------------------------------------------------
    
    listOfPts = []      # List to collect the selected points.
    
    while key & 0xFF != 27:         # Press esc to break.

        imgTemp = np.array( img )      # Temporary image.

        # Displaying all the points in listOfPts on the image.
        for i in range( len(listOfPts) ):
            cv2.circle( imgTemp, tuple(listOfPts[i]), 3, (0, 255, 0), -1 )
            
        # After clicking on the image, press any key (other than esc) to display
        # the point on the image.
        
        if cBix > 0 and cBiy > 0:
            print( '\r{}'.format( ' '*80 ), end='' )   # Erase the last line.
            print( '\rNew point: ({}, {}). Press \'s\' to save.'.format(cBix, cBiy), end='' )
        
            # Since this point is not saved yet, so it is displayed on the 
            # temporary image.
            cv2.circle( imgTemp, (cBix, cBiy), 3, (0, 0, 255), -1 )
            
        cv2.imshow( 'Image', imgTemp )
        key = cv2.waitKey(125)
        
        # If 's' is pressed then the point is saved to the listOfPts.
        if key == ord('s'):
            listOfPts.append( [cBix, cBiy] )
            cv2.circle( imgTemp, (cBix, cBiy), 3, (0, 255, 0), -1 )
            cBix, cBiy = -1, -1
            print( '\nPoint Saved.' )
            
        # Delete point by pressing 'd'.
        elif key == ord('d'):   cBix, cBiy = -1, -1
        
#-------------------------------------------------------------------------------

    # Map the selected points back to the size of original image using the 
    # wRatio and hRatio (if they we resized earlier).
    if resized:   listOfPts = [ [ int( p[0] * wRatio ), int( p[1] * hRatio ) ] \
                                                        for p in listOfPts ]
    return listOfPts

#===============================================================================
        
def timeStamp():
    '''
    Returns the current time stamp including the date and time with as a string 
    of the following format as shown.
    '''
    return datetime.datetime.now().strftime( '_%m_%d_%Y_%H_%M_%S' )

#===============================================================================

prettyTime = lambda x: str( datetime.timedelta( seconds=x ) )

#===============================================================================

def findOneContourIntPt( contourPtArr=None ):
    '''
    This function takes in a contour and returns a point that is garunteed to 
    lie inside the contour (mostly it returns a point near the edge of the contour).
    The input is the array of points which the contours are composed of in opencv.
    '''
    nPts = contourPtArr.shape[0]
    
    for i in range( nPts ):      # Scanning all the points in the contour.
        # Contours are arrays with dimensions (nPts, 1, 2).
        # But the format of points in this contour array is [x, y], 
        # not [y, x] like other numpy arrays, because it is created by opencv.
        x, y = contourPtArr[i, 0, 0], contourPtArr[i, 0, 1]
        
        # Now x, y is a boundary point of this contour. So this means that 
        # some of the neighboring points of this x, y should lie inside the contour.
        # So all 9 points in the neighborhood of x, y (including x, y) is checked.
        for x1 in range( x-5, x+5 ):
            for y1 in range( y-5, y+5 ):
                test = cv2.pointPolygonTest( contourPtArr, (x1, y1), measureDist=False )
                #print( 'test', test )
                if int(test) > 0:   # Positive output means point is inside contour.
                    return [x1, y1]
                
                #If all these are outside the contour (which is theoritically 
                #impossible), only then the next point in the contour is checked.
                
        return [x, y]

#===============================================================================

def findLatestCkpt( checkpointDirPath=None, training=True ):
    '''
    Finds out the latest checkpoint file in the checkpoint directory and
    deletes the incompletely created checkpoint.
    It returns the metaFilePath and ckptPath if found, else returns None.
    It also returns the epoch number of the latest completed epoch.
    The usual tensorflow funtion used to find the latest checkpoint does not
    take into account the fact that the learning rate or the batch size for the 
    training may have changed since the last checkpoint. So for that this 
    function and the json file created along with the checkpoint are used to 
    find the true latest checkpoint.
    It returns the checkpoint and json filepath.
    '''
    if checkpointDirPath is None:
        print( '\nERROR: one or more input arguments missing ' \
               'in findLatestCkpt. Aborting.\n' )
        sys.exit()
    
    # If this is a testing mode, and no checkpoint directory is there, then abort.
    if not os.path.exists( checkpointDirPath ) and not training:
        print( '\nERROR: checkpoint directory \'{}\' not found. ' \
               'in findLatestCkpt. Aborting.\n'.format( checkpointDirPath ) )
        sys.exit()

#-------------------------------------------------------------------------------

    # Create a folder to store the model checkpoints.
    if not os.path.exists( checkpointDirPath ):  # If no previous model is saved.
        os.makedirs( checkpointDirPath )
        return None, None, 0
    
    # If there is previously saved model, then import the graph 
    # along with all the variables, operations etc. (.meta file).
    # Import the variable values (.data binary file with the use of
    # the .index file as well).
    # There is also a file called 'checkpoint' which keeps a record
    # of the latest checkpoint files saved.
    # There may be other files like '.thumbs' etc. which pops up often in 
    # windows.
    # So all files which do not have the proper extensions are deleted.
    listOfFiles = os.listdir( checkpointDirPath )

    for i in listOfFiles:
        extension = i.split('.')[-1]
        if extension != 'json' and extension != 'meta' and extension != 'index' and \
           extension != 'data-00000-of-00001' and extension != 'checkpoint':
               os.remove( os.path.join( checkpointDirPath, i ) )
    
#-------------------------------------------------------------------------------

    # Name of checkpoint to be loaded (in general the latest one).
    # Sometimes due to keyboard interrupts or due to error in saving
    # checkpoints, not all the .meta, .index or .data files are saved.
    # So before loading the checkpoint we need to check if all the 
    # required files are there or not, else the latest complete 
    # checkpoint files should be loaded. And all the incomplete latest 
    # but incomplete ones should be deleted.

    # List to hold the names of checkpoints which have all files.
    listOfValidCkptPaths = []
    
    listOfFiles = os.listdir( checkpointDirPath )
    
    # If there are files inside the checkpoint directory.
    while len( listOfFiles ) > 0:
        # Continue till all the files are scanned.
        
        fileName = listOfFiles[-1]
        if fileName == 'checkpoint':    listOfFiles.remove( fileName )

        ckptName = '.'.join( fileName.split('.')[:-1] )
        metaFileName = ckptName + '.meta'
        indexFileName = ckptName + '.index'
        dataFileName = ckptName + '.data-00000-of-00001'
        jsonFileName = ckptName + '.json'
        
        ckptPath = os.path.join( checkpointDirPath, ckptName )
        metaFilePath = os.path.join( checkpointDirPath, metaFileName )
        indexFilePath = os.path.join( checkpointDirPath, indexFileName )
        dataFilePath = os.path.join( checkpointDirPath, dataFileName )
        jsonFilePath = os.path.join( checkpointDirPath, jsonFileName )
        
        if metaFileName in listOfFiles and dataFileName in listOfFiles and \
           indexFileName in listOfFiles and jsonFileName in listOfFiles:
                # All the files exists, then this is a valid checkpoint. So 
                # adding that into the listOfValidCkptPaths.
                listOfValidCkptPaths.append( ckptPath )

                # Now removing these files from the listOfFiles as all processing 
                # related to them are done.
                listOfFiles.remove( metaFileName )
                listOfFiles.remove( indexFileName )
                listOfFiles.remove( dataFileName )
                listOfFiles.remove( jsonFileName )

        else:
            # If one or more of the .meta, .index or .data files are 
            # missing, then the remaining are deleted and also removed 
            # from the listOfFiles and then we loop back again.
            if os.path.exists( metaFilePath ):
                os.remove( metaFilePath )
                listOfFiles.remove( metaFileName )
            if os.path.exists( indexFilePath ):
                os.remove( indexFilePath )
                listOfFiles.remove( indexFileName )
            if os.path.exists( dataFilePath ):
                os.remove( dataFilePath )
                listOfFiles.remove( dataFileName )
            if os.path.exists( jsonFilePath ):
                os.remove( jsonFilePath )
                listOfFiles.remove( jsonFileName )

        #print( len(listOfFiles) )

#-------------------------------------------------------------------------------

    # At this stage we do not have any incomplete checkpoints in the
    # checkpointDirPath. So now we find the latest checkpoint.
    latestCkptIdx, latestCkptPath = 0, None
    for ckptPath in listOfValidCkptPaths:
        currentCkptIdx = ckptPath.split('-')[-1]   # Extract checkpoint index.
        
        # If the current checkpoint index is '', (which can happen if the
        # checkpoints are simple names like 'cnn_model' and do not have 
        # index like cnn_model.ckpt-2 etc.) then break.
        if currentCkptIdx == '':    break
        
        currentCkptIdx = int( currentCkptIdx )
        
        if currentCkptIdx > latestCkptIdx:     # Compare.
            latestCkptIdx, latestCkptPath = currentCkptIdx, ckptPath
            
    # This will give the latest epoch that has completed successfully.
    # When the checkpoints are saved the epoch is added with +1 in the 
    # filename. So for extracting the epoch the -1 is done.
    latestEpoch = latestCkptIdx if latestCkptIdx > 0 else 0
    
##-------------------------------------------------------------------------------

    ##latestCkptPath = tf.train.latest_checkpoint( checkpointDirPath )
    # We do not use the tf.train.latest_checkpoint( checkpointDirPath ) 
    # function here as it is only dependent on the 'checkpoint' file 
    # inside checkpointDirPath. 
    # So this does not work properly if the latest checkpoint mentioned
    # inside this file is deleted because of incompleteness (missing 
    # some files).

    #ckptPath = os.path.join( checkpointDirPath, 'tiny_yolo.ckpt-0' )

#-------------------------------------------------------------------------------

    if latestCkptPath != None:
        # This will happen when only the 'checkpoint' file remains.
        #print( latestCkptPath )
        latestJsonFilePath = latestCkptPath + '.json'
        return latestJsonFilePath, latestCkptPath, latestEpoch
    
    else:   
        # If no latest checkpoint is found or all are deleted 
        # because of incompleteness and only the 'checkpoint' file 
        # remains, then None is returned.
        return None, None, 0

#===============================================================================

def datasetMeanStd( dataDir=None ):
    '''
    Takes in the location of the images as input.
    Calculates the mean and std of the images of a dataset that is needed 
    to normalize the images before training.
    Returns the mean and std in the form of float arrays 
    (e.g. mean = [ 0.52, 0.45, 0.583 ], std = [ 0.026, 0.03, 0.0434 ] )
    '''
    imgDir = os.path.join( dataDir, 'images' )
    listOfImg = os.listdir( imgDir )
    meanOfImg = np.zeros( ( inImgH, inImgW, 3 ), dtype=np.float32 )
    meanOfImgSquare = np.zeros( ( inImgH, inImgW, 3 ), dtype=np.float32 )
    nImg = len( listOfImg )
    
    for idx, i in enumerate(listOfImg):
        img = cv2.imread( os.path.join( imgDir, i ) )
        
        print( '\rAdding the images to create mean and std {} of {}'.format( idx+1, \
                                                len(listOfImg) ), end = '' )
        meanOfImg += img / nImg
        meanOfImgSquare += img * ( img / nImg )
    
    # Now taking mean of all pixels in the mean image created in the loop.
    # Now meanOfImg is 224 x 224 x 3.
    meanOfImg = np.mean( meanOfImg, axis=0 )
    meanOfImgSquare = np.mean( meanOfImgSquare, axis=0 )
    # Now meanOfImg is 224 x 3.
    meanOfImg = np.mean( meanOfImg, axis=0 )
    meanOfImgSquare = np.mean( meanOfImgSquare, axis=0 )
    # Now meanOfImg is 3.
    variance = meanOfImgSquare - meanOfImg * meanOfImg
    std = np.sqrt( variance )
    
    return meanOfImg, std

#===============================================================================
        
def getImgLabel( curLoc, imgName ):
    '''
    This function takes in the location of an current folder (train or test or
    valid, which contains the images folder) and the image file name, and then 
    accesses the json file for this image and returns the details from that file
    as a list.
    If the image has one sample, this list will have one subdictionary and if the 
    image has two samples, then this list will have two subdictionaries.
    It also creates the multi hot label for the image (which is a list of 1's and 0's).
    If there is 1 or more Eosinophils: label = [1,0,0,0,0,0,0,0,0,0]
    If there is 1 or more Eosinophils and 1 or more Neutrophil: label = [1,0,1,0,0,0,0,0,0,0]
    If there is 1 or more Neutrophils and 1 or more partialWBCs: label = [0,0,1,0,0,1,0,0,0,0]
    '''
    labelName = imgName[:-4] + '.json'
    labelLoc = os.path.join( curLoc, 'labels', labelName )
    
    # Reading the json file.
    with open( labelLoc, 'r' ) as infoFile:
        infoDict = json.load( infoFile )
        
    nObj = len( infoDict )  # Number of objects in the image.

    multiHotLabel = np.zeros( nClasses, dtype=np.int32 )
    
    # List to hold the label dictionaries extracted from the json file.
    labelDictList = []
    
    for i in range( nObj ):
        labelDict = infoDict[ str(i) ]
        classIdx = labelDict[ 'classIdx' ]
        labelDictList.append( labelDict )
        multiHotLabel[ classIdx ] = 1
    
    return labelDictList, multiHotLabel

#===============================================================================
        
def getImgLabelDetection( curLoc, imgName ):
    '''
    This function takes in the location of an current folder (train or test or
    valid, which contains the images folder) and the image file name, and then 
    accesses the json file for this image and returns the details from that file
    as a list.
    If the image has one sample, this list will have one subdictionary and if the 
    image has two samples, then this list will have two subdictionaries.
    It also creates the label required for the detection. The label for each 
    image is 14 x 14 x nAnchors x (nClasses + 5) array.
    The function also returns the multihot label as well, similar to the case of
    classification (this comes in handy if we want to compare the classification
    accuracy during the classification phase and the detection phase).
    The function also returns a list of bounding boxes along with what object 
    category is there within that bounding box, as a 5 element list 
    (classIdx, x, y, w, h). This is needed to find the mAP during testing phase.
    This list will not be of a fixed lenght, as this will vary with the number of
    objects present in the image.
    '''
    labelName = imgName[:-4] + '.json'
    labelLoc = os.path.join( curLoc, 'labels', labelName )
    
    # Reading the json file.
    with open( labelLoc, 'r' ) as infoFile:
        infoDict = json.load( infoFile )
        
    nObj = len( infoDict )  # Number of objects in the image.

    regionLabel = np.zeros( (finalLayerH, finalLayerW, nAnchors, nClasses + 5), \
                                                               dtype=np.float32 )
    
    # Creating the multihot label as well.
    multiHotLabel = np.zeros( nClasses, dtype=np.int32 )
    
    # This is the list of class index and their corresponding bounding boxes.
    # This is blank if no object is present in the image.
    listOfClassIdxAndBbox = []
    
    # List to hold the label dictionaries extracted from the json file.
    labelDictList = []
    
    for i in range( nObj ):
        labelDict = infoDict[ str(i) ]
        classIdx = labelDict[ 'classIdx' ]

        posX, posY = labelDict[ 'posX' ], labelDict[ 'posY' ]
        bboxW, bboxH = labelDict[ 'bboxW' ], labelDict[ 'bboxH' ]
        tlX, tlY = labelDict[ 'tlX' ], labelDict[ 'tlY' ]
        
        multiHotLabel[ classIdx ] = 1
        
        listOfClassIdxAndBbox.append( [ classIdx, tlX, tlY, bboxW, bboxH ] )
        
        # Finding the one hot class label.
        oneHotClassLabel = np.zeros( nClasses ).tolist()
        oneHotClassLabel[ classIdx ] = 1
        
        # Finding the pixel of the final activation layer where the center of 
        # the bbox of the current object will lie. Also finding the offset from 
        # the location (as the location coordinate will be integer values).
        gridX, gridY = (posX / inImgW) * finalLayerW, (posY / inImgH) * finalLayerH
        gridXoffset, gridYoffset = gridX - int(gridX), gridY - int(gridY)
        gridX, gridY = int(gridX), int(gridY)
        
#-------------------------------------------------------------------------------

        # Finding the best anchor boxes which will have good iou score with the
        # bbox of the current object and also findin the scales by which these 
        # anchor boxes have to be scaled to match the ground truth bbox.
        
        # The ground truth box in this case is not the bbox obtained from the 
        # annotation in the image. Because this annotated bbox size is relative
        # to a inImgH x inImgW sized image. But to compare it with the anchor 
        # boxes (whose sizes are relative to the finalLayerH x finalLayerW sized
        # image), it has to be scaled down to the size of the finalLayerH x 
        # finalLayerW sized image.
        resizedBboxW = (bboxW / inImgW) * finalLayerW
        resizedBboxH = (bboxH / inImgH) * finalLayerH
        
        anchorFound = False    # Indicates if suitable anchor boxes are found.
        maxIou, maxIouIdx = 0, 0
        
        for adx, a in enumerate( anchorList ):
            
            # Finding the iou with each of the anchor boxes in the list.
            # This iou is different from the one calculated by the findIOU function.
            # Here the iou is calculated assuming that the center of the ground 
            # truth bbox and the anchor box center coincides.
            minW, minH = min( a[0], resizedBboxW ), min( a[1], resizedBboxH )
            iou = (minW * minH) / (a[0] * a[1] + resizedBboxW * resizedBboxH - minW * minH)
            
            if iou > iouThresh:
                anchorFound = True
                anchorWoffset, anchorHoffset = resizedBboxW / a[0], resizedBboxH / a[1]

                # Store this into the regionLabel in the suitable location.
                regionLabel[ gridY, gridX, adx ] = oneHotClassLabel + \
                                                   [ gridXoffset, gridYoffset, \
                                                     np.log(anchorWoffset), \
                                                     np.log(anchorHoffset), 1.0 ]
                # The 1.0 is the confidence score that an object is present.
            
            # Also keep a record of the best anchor box found.
            if iou > maxIou:    maxIou, maxIouIdx = iou, adx
            
#-------------------------------------------------------------------------------

        # If it happens that none of the anchor boxes have a good enough iou 
        # score (all the scores are less than iouThresh), then just store the 
        # anchor box that has the max iou among all (even though it can be lower
        # than iouThresh). (This may happen when the ground truth bbox is of such
        # a dimension, that none of the anchors is having a proper match with a
        # good iou with it).
        if not anchorFound:
            anchorWoffset = resizedBboxW / anchorList[ maxIouIdx ][0]
            anchorHoffset = resizedBboxH / anchorList[ maxIouIdx ][1]

            # Store this into the regionLabel in the suitable location.
            regionLabel[ gridY, gridX, maxIouIdx ] = oneHotClassLabel + \
                                                     [ gridXoffset, gridYoffset, \
                                                       np.log(anchorWoffset), \
                                                       np.log(anchorHoffset), 1.0 ]
            # The 1.0 is the confidence score that an object is present.
        
#-------------------------------------------------------------------------------
        
        labelDictList.append( labelDict )
    
    return labelDictList, regionLabel, multiHotLabel, listOfClassIdxAndBbox

#===============================================================================
        
def getImgLabelSegmentation( curLoc, imgName ):
    '''
    This function takes in the location of an current folder (train or test or
    valid, which contains the images folder) and the image file name, and then 
    accesses the segment label to create the segment label batch.
    '''

    # Creating the label name from the image name.
    labelName = imgName.split('_')
    labelName.insert( -4, 'seg' )
    labelName = '_'.join( labelName )
    labelLoc = os.path.join( curLoc, 'segments', labelName )
    
    # Reading the segment image file.
    segLabelImg = cv2.imread( labelLoc )
    
    # Creating a weight map. This array should not have a channel axis, otherwise
    # tf.losses.softmax_cross_entropy can handle it.
    segWeightMap = np.zeros( ( inImgH, inImgW ) )
    
    for c in range( nClasses ):
        segColor = np.array( classIdxToColor[c] )
        
        # Creating the mask for every class by using color filter.
        segMask = cv2.inRange( segLabelImg, segColor, segColor ) # Values from 0-255.
        
        # Stack the segMasks along the depths to create the segmentLabel.
        # For 10 classes, the segmentLabel will be an image with 10 channels.
        segmentLabel = np.dstack( (segmentLabel, segMask) ) if c > 0 else segMask
        
        # Updating the weight map.
        segWeightMap += (segMask / 255.0) * classIdxToSegColorWeight[c]
        
    # Since this is basically creating a one hot vector for every pixel, so for 
    # the pixels that are just background, and not part of any object, there 
    # should be a channel as well. Hence a channel is added for the all black 
    # pixels as well. So for 10 classes the number of channels for this segmentLabel
    # array will be 11.
    segColor = np.array( [0,0,0] )
    segMask = cv2.inRange( segLabelImg, segColor, segColor )
    segmentLabel = np.dstack( (segmentLabel, segMask) )
        
    # Updating the weight map.
    segWeightMap += (segMask / 255.0) * classIdxToSegColorWeight[ nClasses ]

    # Reading the json file.
    jsonFileName = imgName[:-4] + '.json'
    jsonFileLoc = os.path.join( curLoc, 'labels', jsonFileName )
    
    with open( jsonFileLoc, 'r' ) as infoFile:
        infoDict = json.load( infoFile )
        
    nObj = len( infoDict )  # Number of objects in the image.

    # List to hold the label dictionaries extracted from the json file.
    labelDictList = []
    
    for i in range( nObj ):
        labelDict = infoDict[ str(i) ]
        classIdx = labelDict[ 'classIdx' ]
        labelDictList.append( labelDict )
    
    return labelDictList, segmentLabel, segWeightMap

#===============================================================================

def calculateSegMapWeights( dataDir=None ):
    '''
    This function calculates the weights of the different segment maps of the 
    different objects in the images of the given directory. This is useful during
    training the segmentation network where because of the disparity in the number
    of pixels of different colors in the segmented map, different weights have 
    to be assigned to them. Otherwise the network will become biased to some  
    particular segment map.
    It also counts the number of objects present in all the images in the given
    dataDir and calculates the average number of pixels used to represent every
    class object.    
    '''
    jsonLoc = os.path.join( dataDir, 'labels' )
    listOfJson = os.listdir( jsonLoc )

    nClassObjs = np.zeros( nClasses )    # Blank array.

    labelLoc = os.path.join( dataDir, 'segments' )
    listOfSegLabel = os.listdir( labelLoc )
    nSegLabels = len( listOfSegLabel )
    
    segLabelPixels = np.zeros( nClasses+1 )    # Blank array.
    pppp = 0
    
    for i in range( nSegLabels ):
        # Reading the dictionary from the json file.
        jsonFileName = listOfJson[i]
        with open( os.path.join( jsonLoc, jsonFileName ), 'r' ) as infoFile:
            infoDict = json.load( infoFile )
        nObjInCurrentImg = np.zeros( nClasses )     # Blank array.
        
        # Counting the number of different objects in the image.
        for k, v in infoDict.items():
            classIdx = v['classIdx']
            nObjInCurrentImg[ classIdx ] += 1
            
        # Adding to the total object count.
        nClassObjs += nObjInCurrentImg
        
#-------------------------------------------------------------------------------

        segLabelName = listOfSegLabel[i]
        segLabelLoc = os.path.join( labelLoc, segLabelName )
        segLabel = cv2.imread( segLabelLoc )    # Reading the segment map image.
        currentSegLabelPixels = np.zeros( nClasses+1 )      # Blank array.
        
        # Now count the number of pixels of each type of segment maps (indicated
        # by a separate color) in the current image.
        for c in range( nClasses ):
            segColor = np.array( classIdxToColor[c] )
            mask = cv2.inRange( segLabel, segColor, segColor )
            mask = mask / 255.0     # Converting from 0-255 to 0-1 range.
            nPixels = np.sum( mask )
            currentSegLabelPixels[c] += nPixels    # Add to pixel count for current class.
            
        # Now count the number of background pixels which are black.
        segColor = np.array( [0,0,0] )
        mask = cv2.inRange( segLabel, segColor, segColor )
        mask = mask / 255.0
        nPixels = np.sum( mask )
        currentSegLabelPixels[ nClasses ] += nPixels
        
        # Adding to the total pixel count.
        segLabelPixels += currentSegLabelPixels
        
        print( '[{}/{}] Total pixels in {}: {}'.format( i+1, nSegLabels, segLabelName, \
                                            np.sum(currentSegLabelPixels) ) )

#-------------------------------------------------------------------------------

    #print( segLabelPixels )
    nTotalPixels = np.sum( segLabelPixels )
    segWeights = nTotalPixels / segLabelPixels      # Calculating the weights.
    
    # Calculating the average number of pixels used to represent every class object.
    # The background class pixels are however ignored.
    avgPixelsPerClassObj = segLabelPixels[ : nClasses ] / nClassObjs
    
    print( 'Segment weights:\n{}\n'.format( segWeights ) )
    print( 'Average number of pixels per object:\n{}\n'.format( avgPixelsPerClassObj ) )
    #print( nClassObjs, np.sum( nClassObjs ) )

#-------------------------------------------------------------------------------

    #for i in range( nSegLabels ):
        #segLabelName = listOfSegLabel[i]
        #segLabelLoc = os.path.join( labelLoc, segLabelName )
        #segLabel = cv2.imread( segLabelLoc )    # Reading the segment map image.
        #segLabel1 = copy.deepcopy( segLabel )
        #listOfColors = [ v for k, v in classIdxToColor.items() ] + [ [0,0,0] ]
        
        #print( segLabelName )
        #exception = []
        
        #for x in range(224):
            #for y in range(224):
                #color = segLabel[ y, x, :].tolist()
                #if color in listOfColors:
                    #pass
                #else:
                    #exception.append( color )
                    #segLabel[y,x,:] = [255,255,255]
                    ##segLabel[y,x,:] = [0,0,0]
                    
        #print( exception )
        #print( len(exception) )
        #cv2.imshow( 'segLabel', segLabel )
        #cv2.imshow( 'original', segLabel1 )
        #cv2.waitKey(0)

#===============================================================================

def createBatchForClassification( dataDir=None, listOfImg=None, batchSize=None, \
                                  shuffle=False, mean=0.0, std=1.0 ):
    '''
    This function takes in a list of images and their location directory of the
    dataset. It also takes in the batch size and returns an image batch, a label 
    batch and the updated listOfImg.
    The bounding box information is there in some json files having the same 
    name as the image files.
    The final batch of images and labels are sent as numpy arrays.
    The list of remaining images and list of selected images are also returned.
    '''
    if dataDir is None or listOfImg is None or batchSize is None:
        print( '\nERROR: one or more input arguments missing ' \
               'in createBatchForClassification. Aborting.\n' )
        sys.exit()

    # Shuffling in place if shuffle flag is True (training phase). 
    # This will be false for test and validation phases.
    if shuffle:     random.shuffle( listOfImg )
    
    listOfBatchImg = listOfImg[ 0 : batchSize ]
    
    imgBatch, labelBatch = [], []
    for i in listOfBatchImg:
        img = cv2.imread( os.path.join( dataDir, 'images', i ) )
        
        labelDictList, multiHotLabel = getImgLabel( dataDir, i )

        labelBatch.append( multiHotLabel )

#        print( multiHotLabel, os.path.join( dataDir, 'images', i ) )
#        cv2.imshow( 'Image', img )
#        cv2.waitKey(0)
        
#        # Normalizing the image by mean and std.
#        img = (img - mean) / std
        
        # Converting image to range -1 to 1.
        # The image is explicitly converted to float32 to match the type 
        # specified in the placeholder. If img would have been directly divided
        # by 127.5, then it would result in np.float64.
        img = np.asarray( img, dtype=np.float32 ) / 127.5 - 1.0
        
        imgBatch.append( img )  
        
    # Removing these from the original listOfImg by first converting them to a 
    # set and then removing the set of element in the imgBatch and converting
    # back the resulting set to list.
    listOfImg = list( set( listOfImg ) - set( listOfBatchImg ) )

    return np.array( imgBatch ), np.array( labelBatch ), listOfImg, listOfBatchImg

#===============================================================================

def createBatchForDetection( dataDir=None, listOfImg=None, batchSize=None, \
                                  shuffle=False, mean=0.0, std=1.0 ):
    '''
    This function takes in a list of images and their location directory of the
    dataset. It also takes in the batch size and returns an image batch, a label 
    batch and the updated listOfImg.
    The bounding box information is there in some json files having the same 
    name as the image files.
    The final batch of images and labels are sent as numpy arrays.
    The list of remaining images and list of selected images are also returned.
    A batch of multihot classification labels are also returned. This is for 
    the convenience of comparing the classification accuracy during detection
    and classification phases.
    A batch of class idx and their corresponding bounding boxes are also returned
    which are used to calculate the mAP during the testing phase.
    '''
    if dataDir is None or listOfImg is None or batchSize is None:
        print( '\nERROR: one or more input arguments missing ' \
               'in createBatchForDetection. Aborting.\n' )
        sys.exit()

    # Shuffling in place if shuffle flag is True (training phase). 
    # This will be false for test and validation phases.
    if shuffle:     random.shuffle( listOfImg )
    
    listOfBatchImg = listOfImg[ 0 : batchSize ]
    
    imgBatch, labelBatch, labelBatchMultiHot, labelBatchClassIdxAndBbox = [], [], [], []
    for i in listOfBatchImg:
        img = cv2.imread( os.path.join( dataDir, 'images', i ) )
        
        labelDictList, regionLabel, multiHotLabel, listOfClassIdxAndBbox = \
                                            getImgLabelDetection( dataDir, i )

        labelBatch.append( regionLabel )
        
        labelBatchMultiHot.append( multiHotLabel )
        
        labelBatchClassIdxAndBbox.append( listOfClassIdxAndBbox )

#        print( multiHotLabel, os.path.join( dataDir, 'images', i ) )
#        cv2.imshow( 'Image', img )
#        cv2.waitKey(0)

#        # Normalizing the image by mean and std.
#        img = (img - mean) / std
        
        # Converting image to range -1 to 1.
        # The image is explicitly converted to float32 to match the type 
        # specified in the placeholder. If img would have been directly divided
        # by 127.5, then it would result in np.float64.
        img = np.asarray( img, dtype=np.float32 ) / 127.5 - 1.0
        
        imgBatch.append( img )        
        
    # Removing these from the original listOfImg by first converting them to a 
    # set and then removing the set of element in the imgBatch and converting
    # back the resulting set to list.
    listOfImg = list( set( listOfImg ) - set( listOfBatchImg ) )

    return np.array( imgBatch ), np.array( labelBatch ), np.array( labelBatchMultiHot ), \
                            labelBatchClassIdxAndBbox, listOfImg, listOfBatchImg

#===============================================================================

def createBatchForSegmentation( dataDir=None, listOfImg=None, batchSize=None, \
                                  shuffle=False, mean=0.0, std=1.0 ):
    '''
    This function takes in a list of images and their location directory of the
    dataset. It also takes in the batch size and returns an image batch, a label 
    batch and the updated listOfImg.
    The label for segmentation is a set of images which has as many channels as 
    the number of classes.
    '''
    if dataDir is None or listOfImg is None or batchSize is None:
        print( '\nERROR: one or more input arguments missing ' \
               'in createBatchForSegmentation. Aborting.\n' )
        sys.exit()

    # Shuffling in place if shuffle flag is True (training phase). 
    # This will be false for test and validation phases.
    if shuffle:     random.shuffle( listOfImg )
    
    listOfBatchImg = listOfImg[ 0 : batchSize ]
    
    imgBatch, labelBatch, weightBatch = [], [], []
    for i in listOfBatchImg:
        img = cv2.imread( os.path.join( dataDir, 'images', i ) )
        
        labelDictList, segmentLabel, segWeightMap = getImgLabelSegmentation( dataDir, i )

        # Converting image to range 0 to 1.
        # The image is explicitly converted to float32 to match the type 
        # specified in the placeholder. If img would have been directly divided
        # by 255.0, then it would result in np.float64.
        segmentLabel = np.asarray( segmentLabel, dtype=np.float32 ) / 255.0

        labelBatch.append( segmentLabel )
        weightBatch.append( segWeightMap )
        
#        print( multiHotLabel, os.path.join( dataDir, 'images', i ) )
#        cv2.imshow( 'Image', img )
#        cv2.waitKey(0)

#        # Normalizing the image by mean and std.
#        img = (img - mean) / std
        
        # Converting image to range -1 to 1.
        # The image is explicitly converted to float32 to match the type 
        # specified in the placeholder. If img would have been directly divided
        # by 127.5, then it would result in np.float64.
        img = np.asarray( img, dtype=np.float32 ) / 127.5 - 1.0
        
        imgBatch.append( img )
    
    # Removing these from the original listOfImg by first converting them to a 
    # set and then removing the set of element in the imgBatch and converting
    # back the resulting set to list.
    listOfImg = list( set( listOfImg ) - set( listOfBatchImg ) )
    
    return np.array( imgBatch ), np.array( labelBatch ), np.array( weightBatch ), \
                            listOfImg, listOfBatchImg

#===============================================================================

def findIOU( boxA, boxB ):
    '''
    Finds the IOU value between two rectangles. The rectangles are described in
    the format of [x, y, w, h], where x and y are the top left corner vertex.
    '''
    xA = max( boxA[0], boxB[0] )      # Max top left x.
    yA = max( boxA[1], boxB[1] )      # Max top left y.
    xB = min( boxA[0] + boxA[2], boxB[0] + boxB[2] )      # Min bottom right x.
    yB = min( boxA[1] + boxA[3], boxB[1] + boxB[3] )      # Min bottom right y.

    # Compute the area of intersection rectangle.
    intersection = max( 0, xB - xA + 1 ) * max( 0, yB - yA + 1 )

    # Compute the area of both the rectangles
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    # Compute the intersection over union by taking the intersection area and 
    # dividing it by the sum of areas of the rectangles - interesection area.
    union = boxAArea + boxBArea - intersection
    iou = intersection / float( union + 0.000001 )

    # return the intersection over union value, intersection value and union value.
    return iou, intersection, union

#===============================================================================
    
def scanWholeImage( img ):
    '''
    This may be an image that is bigger than what the network is trained with
    (which is inImgH x inImgW).
    Hence, it will be divided into several inImgH x inImgW segments to analyze 
    the presence of wbc in the image.
    '''
    imgH, imgW, _ = img.shape
    
    stepH, stepW = int( inImgH / 2 ), int( inImgH / 2 )
    
    # Dividing the images into a grid of inImgH x inImgW cells, and then 
    # converting all those cells into a batch of images.
    # The top left corner coordinates of these grid cells are also stored in lists.
    imgBatch, locList = [], []
    for r in range( 0, imgH, stepH ):
        for c in range( 0, imgW, stepW ):
            cell = img[ r : r + inImgH, c : c + inImgW ]
            if cell.shape[0] == inImgH and cell.shape[1] == inImgW:
                imgBatch.append( cell )
                locList.append( [c, r] )
            
    # Now if the original image size is not a multiple of inImgH x inImgW, then
    # the remaining portion left out on the right and the bottom margins are 
    # included separately into the imgBatch.

    # Including bottom row if height is not a multiple.
    if imgH % inImgH > 0:
        for c in range( 0, imgW, stepW ):
            cell = img[ imgH - inImgH : imgH, c : c + inImgW ]
            if cell.shape[0] == inImgH and cell.shape[1] == inImgW:
                imgBatch.append( cell )
                locList.append( [c, imgH - inImgH] )
            
    # Including right column if width is not a multiple.
    if imgW % inImgW > 0:
        for r in range( 0, imgH, stepH ):
            cell = img[ r : r + inImgH, imgW - inImgW : imgW ]
            if cell.shape[0] == inImgH and cell.shape[1] == inImgW:
                imgBatch.append( cell )
                locList.append( [imgW - inImgW, r] )

    # Including bottom right corner if both height and width are not multiples.
    if imgH % inImgH > 0 and imgW % inImgW > 0:
        cell = img[ imgH - inImgH : imgH, imgW - inImgW : imgW ]
        if cell.shape[0] == inImgH and cell.shape[1] == inImgW:
            imgBatch.append( cell )
            locList.append( [imgW - inImgW, imgH - inImgH] )
            
##-------------------------------------------------------------------------------
#    
#    # Displaying the corners of the grid cells.        
#    for i in range( len( locList ) ):
#        cv2.circle( img, ( locList[i][0], locList[i][1] ), 2, (0,255,0), 2 )
#        
#    cv2.imshow( 'Img', img )
#    cv2.waitKey(0)
#
#    for i in imgBatch:
#        print(i.shape)
#
#-------------------------------------------------------------------------------

    return np.array( imgBatch ), locList

#===============================================================================
    
def filterAndAddRect( rectList=None, rectangle=None ):
    '''
    This function takes in a list of rectangles and also a new rectangle.
    It then checks if this new rectangle has a high IOU with any other rectangle
    or not. If so, then it averages the two and then stores that in the list
    replacing the old one. But if this new rectangle does not have much overlap
    with any other, then it just stores the rectangle as it is. Any new or 
    updated rectangle is always appended to the end of the rectList.
    '''
    if rectList is None or rectangle is None:
        print( '\nERROR: one or more input arguments missing ' \
               'in filterAndAddRect. Aborting.\n' )
        sys.exit()
    
    if len( rectList ) == 0:    # List is empty.
        rectList.append( rectangle )
        return rectList
    
    # Check for the IOU values.
    overlapFound = False
    for rdx, r in enumerate( rectList ):
        iou, intersection, union = findIOU( r, rectangle )
        if iou > 0.5:
            overlapFound = True
            # Take average of the two rectangles.
            x = int( ( r[0] + rectangle[0] ) * 0.5 )
            y = int( ( r[1] + rectangle[1] ) * 0.5 )
            w = int( ( r[2] + rectangle[2] ) * 0.5 )
            h = int( ( r[3] + rectangle[3] ) * 0.5 )
            rectList.pop( rdx )     # Remove the old rectangle.
            rectList.append( [x,y,w,h] )    # Append the new one.
            # The new or updated rectangle is always appended to the end of the 
            # list.
            
            # Since only one rectangle is added at a time, and since as soon as
            # an overlap is found, the rectangle in the original list os replaced,
            # hence no rectangle already stored in the rectList will have any 
            # overlap with each other. And hence the new rectangle can also have
            # an overlap with only one of these rectangles in the rectList.
            
    if not overlapFound:
        rectList.append( rectangle )
        
    return rectList

#===============================================================================

def localizeWeakly( gapLayer, inferPredLabelList, img=None ):
    '''
    This function weakly localizes the objects based on the output of the conv
    layer just before the global average pooling (gap) layer. It also takes 
    help of the inferPredLabels. Only the output corresponding to a SINGLE image
    can be processed by this function. Not a batch of images.
    But the number of elements in the one or multi hot vectors in the final 
    labels, should be the same as the number of channels
    in this conv layer (which is given as input argument to this function).
    The inferPredLabelList is also the predicted label for one image in the batch
    not the overall inferPredLabel for the entire batch.
    '''

#    layer = inferLayerOut['conv19']
    h, w, nChan = gapLayer.shape
    if type( inferPredLabelList ) != list:
        inferPredLabelList = inferPredLabelList.tolist()
    
    if len( inferPredLabelList ) != nChan:
        print( '\nERROR: the number of elements in the one/multi hot label is ' \
               'not the same as the number of channels in the conv layer input ' \
               'input to this function localizeWeakly. Aborting.\n' )
        return
            
    # Stacking the channels of this layer together for displaying.
    # Also creating a List that will hold information about the labels, centers 
    # and bboxs.
    classAndLocList = []

    for c in range( nChan ):
        channel = gapLayer[:,:,c]
        resizedChan = cv2.resize( channel, (inImgW, inImgW), \
                                     interpolation=cv2.INTER_LINEAR )
        minVal, maxVal = np.amin( resizedChan ), np.amax( resizedChan )

        # Normalizing the output and scaling to 0 to 255 range.
        normalizedChan = ( resizedChan - minVal ) / ( maxVal - minVal \
                                                      + 0.000001 ) * 255
        normalizedChan = np.asarray( normalizedChan, dtype=np.uint8 )
        
        # Stacking the normalized channels.
        layerImg = normalizedChan if c == 0 else \
                                np.hstack( ( layerImg, normalizedChan ) )
                
#-------------------------------------------------------------------------------

        # WEAK LOCALIZATION
        
        # If there is a 1 in the predicted label, the corresponding 
        # channel of the gap layer is used to weakly localize the wbc cell.
        if inferPredLabelList[c]:
            # The normalized image is subjected to otsu's thresholding.
            _, binaryImg = cv2.threshold( normalizedChan, 0, 255, \
                                          cv2.THRESH_BINARY+cv2.THRESH_OTSU )
                    
            # Contours are then found out from the thresholded binary image,
            # after appending borders to the image.
            binaryImg = cv2.copyMakeBorder( binaryImg, 5,5,5,5, \
                                            cv2.BORDER_CONSTANT, value=0 )
            returnedTuple = cv2.findContours( binaryImg, mode=cv2.RETR_TREE, \
                                               method=cv2.CHAIN_APPROX_SIMPLE )
            contours = returnedTuple[-2]
            
            # Locate the center of these contours with a mark.
            rectList = []
            for cdx, cont in enumerate( contours ):
                cont = cont - 5    # Offsetting the border thickness.

                # Finding the bounding rectangle.
                x, y, w, h = cv2.boundingRect( cont )
                cx, cy = int(x+w/2), int(y+h/2)    # Center of the contour.
                
                # Store these bounding rectangles in a list as well, 
                # after checking that there is no duplicate or similar
                # rectangle. In other words, do a non-maximum supression
                # and then save the rectangles.
                rectList = filterAndAddRect( rectList, [cx-50,cy-50,100,100] )
                
                # Storing the contours after offsetting border thickness.
                contours[ cdx ] = cont

            # Storing the information about the labels, centers and bboxs in the
            # classAndLocList.
            for rectangle in rectList:
                x, y, w, h = rectangle
                classAndLocList.append( [ c, int(x+w/2), int(y+h/2), x, y, w, h ] )

##-------------------------------------------------------------------------------
#
#            if img is None:     continue
#        
#            # Draw the contours.
#            cv2.drawContours( img, contours, -1, (0,255,255), 2 )
#        
#            # Now draw the rectangles.
#            predName = classIdxToName[c]
#            
#            for rectangle in rectList:
#                x, y, w, h = rectangle
#                cx, cy = int(x+w/2), int(y+h/2)    # Center of the contour.
#                cv2.circle( img, (cx, cy), 1, (255,255,0), 1 )
#                cv2.rectangle( img, (cx-50, cy-50), (cx+50, cy+50), (255,255,0), 2 )
#                cv2.putText( img, predName[0], (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, \
#                                                0.6, (0,255,0), 2, cv2.LINE_AA )
#
#    if img is not None:
#        cv2.imshow( 'Weak localization', img )
#        cv2.imshow( 'Gap layer', layerImg )
#        cv2.waitKey(0)                        
#                                    
#-------------------------------------------------------------------------------

    return classAndLocList
        
#===============================================================================

def nonMaxSuppression( predResult ):
    '''
    This function takes in the raw output of the network and then performs a 
    non-maximum suppression on the results to filter out the redundant and 
    unwanted bounding boxes.
    The filtered boxes are in the format of (top-left-X, top-left-Y, width, height).
    '''
    # nResults is the number of image results present in this data.
    nResults, _, _, _, _ = predResult.shape
    
    detectedBatchClassScores, detectedBatchClassIdxes, detectedBatchClassNames, \
                                            detectedBatchBboxes = [], [], [], []
    
    for b in range( nResults ):
        classes, bboxes = [], []

        for i in range( finalLayerH ):
            for j in range( finalLayerW ):
                for a in range( nAnchors ):
                    oneHotVec = predResult[ b, i, j, a, : nClasses ]
                    xOffset = predResult[ b, i, j, a, -5 ]
                    yOffset = predResult[ b, i, j, a, -4 ]
                    wOffset = predResult[ b, i, j, a, -3 ]
                    hOffset = predResult[ b, i, j, a, -2 ]
                    confOffset = predResult[ b, i, j, a, -1 ]
                    
                    classProb = oneHotVec * confOffset
                    
                    # Now rescaling the w and h to the actual input image size.
                    w = ( wOffset / finalLayerW ) * inImgW
                    h = ( hOffset / finalLayerH ) * inImgH
                    
                    # Calculating the top left corner coordinates of the boxes.
                    x = ( ( j + xOffset ) / finalLayerW ) * inImgW
                    y = ( ( i + yOffset ) / finalLayerH ) * inImgH
                    tlX, tlY = x - w * 0.5, y - h * 0.5
                    
                    # Recording the bboxes and classes into the lists.
                    bboxes.append( [ int(tlX), int(tlY), int(w), int(h) ] )
                    classes.append( classProb )
                    
#-------------------------------------------------------------------------------
                    
        # Converting the classes list into array.
        classes = np.array( classes )
        
        # classes and bboxes arrays have one row for each of 
        # finalLayerH x finalLayerW x nAnchors (14 x 14 x 5 = 980) anchor boxes. 
        # And the columns represent probabilities of the nClasses (6) classes. 
        
        # We will transpose the classes array before non max suppression. So 
        # now there is one column for each anchor box and each row represents 
        # the probability of the classes.
        classes = np.transpose( classes )
        
        for c in range( nClasses ):
            classProb = classes[c]
            
            # Making the class probability 0 if it is less than threshProbDetection.
            classProb = classProb * ( classProb > threshProbDetection )

            # Sorting both the arrays in descending order (arrays can also be 
            # sorted in this manner like lists).
            # In the end after the redundant boxes are removed, these sorted 
            # lists has to be reverted back to their original (unsorted) form.
            # To do that a list of indexes are also maintained as a record.
            indexes = list( range( len( bboxes ) ) )
            classProbSorted, bboxesSorted, indexes = zip( *sorted( zip( classProb, \
                                                          bboxes, indexes ), \
                                                          key=lambda x: x[0], \
                                                          reverse=True ) )
            
            # The classProbSorted and bboxesSorted are returned as tuples.
            # Converting them to lists.
            classProbSorted, bboxesSorted, indexes = list( classProbSorted ), \
                                                     list( bboxesSorted ), list(indexes)
            
#-------------------------------------------------------------------------------
            
            # Now we are comparing all the boxes for the current class c for 
            # removing redundant ones.
            for bboxMaxIdx in range( len( bboxesSorted ) ):
                # Skipping the boxes if the corresponding class probability is 0.
                # Since the numbers are in floats, so we do not use == 0 here.
                # Instead we compare whether the number is very close to 0 or not.
                if classProbSorted[ bboxMaxIdx ] < 0.000001:      continue
                
                bboxMax = bboxesSorted[ bboxMaxIdx ]    # Box with max class probability.
                
                for bboxCurIdx in range( bboxMaxIdx+1, len( bboxesSorted ) ):
                    # Skipping the boxes if the corresponding class probability is 0.
                    # Since the numbers are in floats, so we do not use == 0 here.
                    # Instead we compare whether the number is very close to 0 or not.
                    if classProbSorted[ bboxCurIdx ] < 0.000001:      continue
                    
                    # Box other than the max class probability box.
                    bboxCur = bboxesSorted[ bboxCurIdx ]
                    
                    # If the iou between the boxes with max probability and the
                    # current box is greater than the iouTh, then that means 
                    # the current box is redundant. So set corresponding class
                    # probability to 0.
                    iou, _, _ = findIOU( bboxMax, bboxCur )
                    if iou > iouThresh:     classProbSorted[ bboxCurIdx ] = 0
            
            # Now that all the redundancy is removed, we restore the probability 
            # values to the original classes array after converting them into 
            # their original (unsorted) format.
            classProbUnsorted, indexes = zip( *sorted( zip( classProbSorted, \
                                                       indexes ), key=lambda x: x[1] ) )

#-------------------------------------------------------------------------------

            # The classProbSorted and bboxesSorted are returned as tuples.
            # Converting them to lists.
            classProbUnsorted, indexes = list( classProbUnsorted ), list(indexes)
            classes[c] = classProbUnsorted

#-------------------------------------------------------------------------------

        # Scanning the cols (each col has class probabilities of an anchor box).
        detectedIdxes, detectedScores, detectedObjs, detectedBboxes = [], [], [], []
        for a in range( len( bboxes ) ):
            maxScore, maxScoreIdx = np.max( classes[:, a] ), np.argmax( classes[:, a] )
            if maxScore > 0:
                detectedScores.append( maxScore )
                detectedIdxes.append( maxScoreIdx )
                detectedObjs.append( classIdxToName[ maxScoreIdx ] )
                detectedBboxes.append( bboxes[ a ] )

        # Recording the bboxes and classes into the list for the entire batch.
        detectedBatchClassScores.append( detectedScores )
        detectedBatchClassIdxes.append( detectedIdxes )
        detectedBatchClassNames.append( detectedObjs )
        detectedBatchBboxes.append( detectedBboxes )

#-------------------------------------------------------------------------------
    
    detectedBatchClassScores = np.array( detectedBatchClassScores )
    detectedBatchClassIdxes = np.array( detectedBatchClassIdxes )
    detectedBatchClassNames = np.array( detectedBatchClassNames )
    detectedBatchBboxes = np.array( detectedBatchBboxes )
        
    return detectedBatchClassScores, detectedBatchClassIdxes, \
                                detectedBatchClassNames, detectedBatchBboxes
    
#===============================================================================

def calculateMAP( allTestMultiHot=None, allTestClassIdxAndBbox=None, \
                  allDetectedClassIdxes=None, allDetectedClassScores=None, \
                  allDetectedBboxes=None ):
    '''
    This function takes in lists of all the details of prediction over an
    entire dataset along with the details of the ground truth indexes and 
    bounding boxes and then calculates the mean average precision over this
    dataset. It also takes in the list of all multihot labels to know which of 
    the images has which kind of objects. This is important to know the average
    precision of the individual classes. All these average precisions are 
    combined together to calculate the mAP.
    '''
    if allTestMultiHot is None or allTestClassIdxAndBbox is None or \
       allDetectedClassIdxes is None or allDetectedClassScores is None or \
       allDetectedBboxes is None:
            print( '\nERROR: one or more input arguments missing ' \
                   'in calculateMAP. Aborting.\n' )
            sys.exit()

    nImgs = len( allTestClassIdxAndBbox )
    mAP = 0
    APlist = []
    
#-------------------------------------------------------------------------------
    
    # Calculating the average precision of each class and then adding them to 
    # find the mAP
    for c in range( nClasses ):
        AP = 0
        nInstance = 0   # Number of instances of this class in the entire dataset.
        
        # Now scanning all the records of the images inside the input lists and
        # creating  a list of true and false positives and scores.
        fullTPlist, fullFPlist, fullScoreList = [], [], []
        
        for i in range( nImgs ):
            multiHotLabel = allTestMultiHot[i]
            
            # Now checking if the multihot label has the object of class c or not.
            # If not then this image is skipped. This is determined by checking
            # if the c-th element of the multiHotLabel is 1 or not.
            if multiHotLabel[c] == 0:   continue
        
            trueClassIdxAndBbox = allTestClassIdxAndBbox[i]
            detectedClassIdxes = allDetectedClassIdxes[i]
            detectedClassScores = allDetectedClassScores[i]
            detectedBboxes = allDetectedBboxes[i]
            
#-------------------------------------------------------------------------------

            # Scanning all the predictions for this image i.
            
            # First count how many of the predicted boxes also predicts class c.
            # Also, store the indexes of these predicted boxes in a list.
            # Create a blank list equal to this count. These will store
            # the true positive status of the predicted boxes.
            indexes = [ kdx for kdx, k in enumerate( detectedClassIdxes ) if k == c ]
            TPlist = np.zeros( len( indexes ), dtype=int ).tolist()
            scoreList = [ detectedClassScores[kdx] for kdx, k in \
                                     enumerate( detectedClassIdxes ) if k == c ]

#-------------------------------------------------------------------------------

            # Now scan through all the records of this image.
            for j in range( len( trueClassIdxAndBbox ) ):
                classIdx, tlX, tlY, bboxW, bboxH = trueClassIdxAndBbox[j]
                
                # Check if the jth record has the object c or not.
                if classIdx != c:       continue

                nInstance += 1    # Counting number of instances of class c.

                bestIOU = iouThreshForMAPcalculation
                bestPdx = -1    # This will become the index of the box with best iou.
                
#-------------------------------------------------------------------------------

                # Now taking only the boxes which are recorded in the indexes list
                # (as only those boxes are detecting the object of class c).
                for pdx, p in enumerate( indexes ):
                    predTlX, predTlY, predBboxW, predBboxH = detectedBboxes[p]
                    
                    # Find the iou now.
                    iou, _, _ = findIOU( [ tlX, tlY, bboxW, bboxH ], \
                                         [ predTlX, predTlY, predBboxW, predBboxH ] )
                    
                    # It may happen that there are multiple bounding boxes which
                    # overlap with the same object. In that case select the one
                    # which has the highest iou score as the best one.
                    if iou > bestIOU:   bestIOU, bestPdx = iou, pdx
                    
#-------------------------------------------------------------------------------
                    
                # Now make this box corresponding to the bestIOU as a true positive.
                # If however the bestPdx is still -1, then it implies that there
                # are no good boxes here. Hence skip the update then.
                if bestPdx > -1:    TPlist[ bestPdx ] = 1
                
            # Now make all the other remaining boxes as false positive.
            FPlist = [ 0 if m == 1 else 1 for m in TPlist ]
            
#-------------------------------------------------------------------------------
             
            # Combining the true and false positive and the score lists into the
            # bigger list.
            fullTPlist += TPlist
            fullFPlist += FPlist
            fullScoreList += scoreList
        
#-------------------------------------------------------------------------------
        
        # Now sort the lists as per the score values.
        sortedScoreList, sortedTPlist, sortedFPlist = zip( *sorted( zip( fullScoreList, \
                                                        fullTPlist, fullFPlist ), \
                                                        key=lambda x: x[0], reverse=True ) )
        
        # The sortedScoreList, sortedTPlist and sortedFPlist are returned as tuples.
        # Converting them to arrays.
        sortedScoreList, sortedTPlist, sortedFPlist = np.array( sortedScoreList ), \
                                                      np.array( sortedTPlist ), \
                                                      np.array( sortedFPlist )

        # Creating the accumulated true and false positive lists.
        accumulatedTP, accumulatedFP = np.cumsum( sortedTPlist ), np.cumsum( sortedFPlist )
        precision = accumulatedTP / ( accumulatedTP + accumulatedFP )
        recall = accumulatedTP / nInstance
        
        # Converting the precision and recall from arrays to list.
        precision, recall = precision.tolist(), recall.tolist()

#-------------------------------------------------------------------------------

        # Calculating the average precision of this class c.
        
#        plt.plot( recall, precision )
#        plt.show()

        # A lot of the recall values evaluated in this manner will be repeated.
        # So taking a set of the distinct recall values, (then sorting them, as 
        # the values may not be in sorted form while creating the set) and taking 
        # the corresponding precision values.
        recallSet = set( recall )
        recallSet = sorted( list( recallSet ) )
        precisionSet = [ precision[ recall.index(r) ] for r in recallSet ]
        
        # The precisionSet now has the precision values which are the vertices of the
        # sawtooth shaped precision recall curve.
        # Sorting the precisionSet in descending order to find the tallest vertices.
        precisionSet, recallSet = zip( *sorted( zip( precisionSet, recallSet ), \
                                               key=lambda x: x[0], reverse=True ) )
    
        # The precisionSet and recallSet are returned as tuples. Converting them to lists.
        precisionSet, recallSet = list( precisionSet ), list( recallSet )
    
        # Appending a 0 to the recallSet.
        recallSet = [0.0] + recallSet
    
#-------------------------------------------------------------------------------

        totalArea, previousStep = 0.0, 0
        for r in range( 1, len(recallSet) ):
            # Calculating the base of the rectangular section.
            base = recallSet[r] - recallSet[ previousStep ]
            
            if base > 0:
                # Calculating the height of the rectangular section.
                height = precisionSet[ r-1 ]
                totalArea += height * base            
                previousStep = r    # Updating the previousStep.
    
        AP = totalArea * 100
        APlist.append( AP )
        mAP += (AP / nClasses)
    
#-------------------------------------------------------------------------------

    return mAP, APlist







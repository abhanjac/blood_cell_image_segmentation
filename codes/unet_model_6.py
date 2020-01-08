# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 11:55:25 2019

This script describes a class for the tiny yolo network.

@author: abhanjac
"""

import tensorflow as tf

from utils_2 import *

class unet1( object ):
    '''
    Class that defines the unet model, along with its associated functions.
    '''
    def __init__(self):
        '''
        Initializes some of the fixed parameters of the model.
        '''
        # Defining methods to initilize the model weights and biases.
        self.initW = tf.glorot_normal_initializer( dtype=tf.float32 )
        self.initB = tf.zeros_initializer()
        
        # Dictionary to hold the individual outputs of each model layer.
        self.layerOut = {}
        # Flag that indicates if the output of the layers has to be saved in 
        # the dictionary or not.
        self.saveLayer = False
        
        # Defining the optimizer.
        # This is done here because we will be needing the optimizerName in the 
        # test function as well. If we will define the optimizer in the train
        # function, then during testing when the train function is not called,
        # the optimizerName will not be initialized. So it is defined in init
        # such that it gets defined as the class object is initialized.
        self.optimizer = tf.train.AdamOptimizer( learning_rate=learningRate )
        # Name of optimizer ('Adam' in this case).
        self.optimizerName = self.optimizer.get_name()
        
        # This flag indicates if the network is in training mode or not.
        self.isTraining = False
        
#===============================================================================
        
    def model(self, x):
        '''
        This defines the overall network structure of the tiny yolo network.

        layer     filters   kernel      input                   output
        0 conv    64        3 x 3 / 1   224 x 224 x 3      ->   224 x 224 x 32
        1 conv    64        3 x 3 / 1   224 x 224 x 32     ->   224 x 224 x 32
        2 max               2 x 2 / 2   224 x 224 x 32     ->   112 x 112 x 32
        3 conv    128       3 x 3 / 1   112 x 112 x 32     ->   112 x 112 x 64
        4 conv    128       3 x 3 / 1   112 x 112 x 64     ->   112 x 112 x 64
        5 max               2 x 2 / 2   112 x 112 x 64     ->   56 x 56 x 64
        6 conv    256       3 x 3 / 1   56 x 56 x 64       ->   56 x 56 x 128
        7 conv    256       3 x 3 / 1   56 x 56 x 128      ->   56 x 56 x 128
        8 max               2 x 2 / 2   56 x 56 x 128      ->   28 x 28 x 128
        9 conv    512       3 x 3 / 1   28 x 28 x 128      ->   28 x 28 x 256
        10 conv   512       3 x 3 / 1   28 x 28 x 256      ->   28 x 28 x 256
        11 max              2 x 2 / 2   28 x 28 x 256      ->   14 x 14 x 256
        12 conv   1024      3 x 3 / 1   14 x 14 x 256      ->   14 x 14 x 512
        13 conv   1024      3 x 3 / 1   14 x 14 x 512      ->   14 x 14 x 512
        14 trconv 512       2 x 2 / 2   14 x 14 x 512      ->   28 x 28 x 256
           concat 10 + 14 = (28 x 28 x 256) * 2            ->   28 x 28 x 512
        15 conv   512       3 x 3 / 1   28 x 28 x 512      ->   28 x 28 x 256
        16 conv   512       3 x 3 / 1   28 x 28 x 256      ->   28 x 28 x 256
        17 trconv 256       2 x 2 / 2   28 x 28 x 256      ->   56 x 56 x 128
           concat 7 + 17  = (56 x 56 x 128) * 2            ->   56 x 56 x 256
        18 conv   256       3 x 3 / 1   56 x 56 x 256      ->   56 x 56 x 128
        19 conv   256       3 x 3 / 1   56 x 56 x 128      ->   56 x 56 x 128
        20 trconv 128       2 x 2 / 2   56 x 56 x 128      ->   112 x 112 x 64
           concat 4 + 20  = (112 x 112 x 64) * 2           ->   112 x 112 x 128
        21 conv   128       3 x 3 / 1   112 x 112 x 128    ->   112 x 112 x 64
        22 conv   128       3 x 3 / 1   112 x 112 x 64     ->   112 x 112 x 64
        23 trconv 64        2 x 2 / 2   112 x 112 x 64     ->   224 x 224 x 32
           concat 1 + 23  = (224 x 224 x 32) * 2           ->   224 x 224 x 64
        24 conv   64        3 x 3 / 1   224 x 224 x 64     ->   224 x 224 x 32
        25 conv   64        3 x 3 / 1   224 x 224 x 32     ->   224 x 224 x 32
        26 conv nClasses+1  1 x 1 / 1   224 x 224 x 32     ->   224 x 224 x nClasses+1
                   
        '''

        x = tf.convert_to_tensor( x, dtype=tf.float32 )
        
        # Input size 224 x 224 x 3 (H x W x D).
        layerIdx = '0'
        layerName = 'conv' + layerIdx
        x0 = tf.layers.conv2d( x, kernel_size=(3,3), filters=32, padding='SAME', \
                              name=layerName, bias_initializer=self.initB, \
                              kernel_initializer=self.initW, use_bias=True, trainable=True )
        if self.saveLayer:  self.layerOut[ layerName ] = x0
        
        layerName = 'activation' + layerIdx
        #x0 = tf.nn.relu( x0, name=layerName )
        x0 = tf.nn.leaky_relu( x0, alpha=leak, name=layerName )
        if self.saveLayer:  self.layerOut[ layerName ] = x0
        
        # Output size 224 x 224 x 32 (H x W x D).
        
#-------------------------------------------------------------------------------
        
        # Input size 224 x 224 x 32 (H x W x D).
        layerIdx = '1'
        layerName = 'conv' + layerIdx
        x1 = tf.layers.conv2d( x0, kernel_size=(3,3), filters=32, padding='SAME', \
                              name=layerName, bias_initializer=self.initB, \
                              kernel_initializer=self.initW, use_bias=True, trainable=True )
        if self.saveLayer:  self.layerOut[ layerName ] = x1
        
        layerName = 'activation' + layerIdx
        #x1 = tf.nn.relu( x1, name=layerName )
        x1 = tf.nn.leaky_relu( x1, alpha=leak, name=layerName )
        if self.saveLayer:  self.layerOut[ layerName ] = x1

        # Output size 224 x 224 x 32 (H x W x D).
        
#-------------------------------------------------------------------------------

        # Input size 224 x 224 x 32 (H x W x D).
        layerIdx = '2'
        layerName = 'pooling' + layerIdx
        x2 = tf.layers.max_pooling2d( x1, pool_size=(2,2), strides=2, \
                                        padding='SAME', name=layerName )
        if self.saveLayer:  self.layerOut[ layerName ] = x2
        
        # Output size 112 x 112 x 32 (H x W x D).
        
#-------------------------------------------------------------------------------
        
        # Input size 112 x 112 x 32 (H x W x D).
        layerIdx = '3'
        layerName = 'conv' + layerIdx
        x3 = tf.layers.conv2d( x2, kernel_size=(3,3), filters=64, padding='SAME', \
                              name=layerName, bias_initializer=self.initB, \
                              kernel_initializer=self.initW, use_bias=True, trainable=True )
        if self.saveLayer:  self.layerOut[ layerName ] = x3
        
        layerName = 'activation' + layerIdx
        #x3 = tf.nn.relu( x3, name=layerName )
        x3 = tf.nn.leaky_relu( x3, alpha=leak, name=layerName )
        if self.saveLayer:  self.layerOut[ layerName ] = x3
        
        # Output size 112 x 112 x 64 (H x W x D).
        
#-------------------------------------------------------------------------------
        
        # Input size 112 x 112 x 64 (H x W x D).
        layerIdx = '4'
        layerName = 'conv' + layerIdx
        x4 = tf.layers.conv2d( x3, kernel_size=(3,3), filters=64, padding='SAME', \
                              name=layerName, bias_initializer=self.initB, \
                              kernel_initializer=self.initW, use_bias=True, trainable=True )
        if self.saveLayer:  self.layerOut[ layerName ] = x4
        
        layerName = 'activation' + layerIdx
        #x4 = tf.nn.relu( x4, name=layerName )
        x4 = tf.nn.leaky_relu( x4, alpha=leak, name=layerName )
        if self.saveLayer:  self.layerOut[ layerName ] = x4
        
        # Output size 112 x 112 x 64 (H x W x D).
        
#-------------------------------------------------------------------------------

        # Input size 112 x 112 x 64 (H x W x D).
        layerIdx = '5'
        layerName = 'pooling' + layerIdx
        x5 = tf.layers.max_pooling2d( x4, pool_size=(2,2), strides=2, \
                                        padding='SAME', name=layerName )
        if self.saveLayer:  self.layerOut[ layerName ] = x5
        
        # Output size 56 x 56 x 64 (H x W x D).
        
#-------------------------------------------------------------------------------
        
        # Input size 56 x 56 x 64 (H x W x D).
        layerIdx = '6'
        layerName = 'conv' + layerIdx
        x6 = tf.layers.conv2d( x5, kernel_size=(3,3), filters=128, padding='SAME', \
                              name=layerName, bias_initializer=self.initB, \
                              kernel_initializer=self.initW, use_bias=True, trainable=True )
        if self.saveLayer:  self.layerOut[ layerName ] = x6
        
        layerName = 'activation' + layerIdx
        #x6 = tf.nn.relu( x6, name=layerName )
        x6 = tf.nn.leaky_relu( x6, alpha=leak, name=layerName )
        if self.saveLayer:  self.layerOut[ layerName ] = x6

        # Output size 56 x 56 x 128 (H x W x D).
        
#-------------------------------------------------------------------------------
        
        # Input size 56 x 56 x 128 (H x W x D).
        layerIdx = '7'
        layerName = 'conv' + layerIdx
        x7 = tf.layers.conv2d( x6, kernel_size=(3,3), filters=128, padding='SAME', \
                              name=layerName, bias_initializer=self.initB, \
                              kernel_initializer=self.initW, use_bias=True, trainable=True )
        if self.saveLayer:  self.layerOut[ layerName ] = x7
        
        layerName = 'activation' + layerIdx
        #x7 = tf.nn.relu( x7, name=layerName )
        x7 = tf.nn.leaky_relu( x7, alpha=leak, name=layerName )
        if self.saveLayer:  self.layerOut[ layerName ] = x7
        
        # Output size 56 x 56 x 128 (H x W x D).
        
#-------------------------------------------------------------------------------

        # Input size 56 x 56 x 128 (H x W x D).
        layerIdx = '8'
        layerName = 'pooling' + layerIdx
        x8 = tf.layers.max_pooling2d( x7, pool_size=(2,2), strides=2, \
                                        padding='SAME', name=layerName )
        if self.saveLayer:  self.layerOut[ layerName ] = x8
        
        # Output size 28 x 28 x 128 (H x W x D).
        
#-------------------------------------------------------------------------------
        
        # Input size 28 x 28 x 128 (H x W x D).
        layerIdx = '9'
        layerName = 'conv' + layerIdx
        x9 = tf.layers.conv2d( x8, kernel_size=(3,3), filters=256, padding='SAME', \
                              name=layerName, bias_initializer=self.initB, \
                              kernel_initializer=self.initW, use_bias=True, trainable=True )
        if self.saveLayer:  self.layerOut[ layerName ] = x9
        
        layerName = 'activation' + layerIdx
        #x9 = tf.nn.relu( x9, name=layerName )
        x9 = tf.nn.leaky_relu( x9, alpha=leak, name=layerName )
        if self.saveLayer:  self.layerOut[ layerName ] = x9
        
        # Output size 28 x 28 x 256 (H x W x D).
        
#-------------------------------------------------------------------------------
        
        # Input size 28 x 28 x 256 (H x W x D).
        layerIdx = '10'
        layerName = 'conv' + layerIdx
        x10 = tf.layers.conv2d( x9, kernel_size=(3,3), filters=256, padding='SAME', \
                              name=layerName, bias_initializer=self.initB, \
                              kernel_initializer=self.initW, use_bias=True, trainable=True )
        if self.saveLayer:  self.layerOut[ layerName ] = x10
        
        layerName = 'activation' + layerIdx
        #x10 = tf.nn.relu( x10, name=layerName )
        x10 = tf.nn.leaky_relu( x10, alpha=leak, name=layerName )
        if self.saveLayer:  self.layerOut[ layerName ] = x10
        
        # Output size 28 x 28 x 256 (H x W x D).
        
#-------------------------------------------------------------------------------

        # Input size 28 x 28 x 256 (H x W x D).
        layerIdx = '11'
        layerName = 'pooling' + layerIdx
        x11 = tf.layers.max_pooling2d( x10, pool_size=(2,2), strides=2, \
                                        padding='SAME', name=layerName )
        if self.saveLayer:  self.layerOut[ layerName ] = x11
        
        # Output size 14 x 14 x 256 (H x W x D).
        
#-------------------------------------------------------------------------------

        # Input size 14 x 14 x 256 (H x W x D).
        layerIdx = '12'
        layerName = 'conv' + layerIdx
        x12 = tf.layers.conv2d( x11, kernel_size=(3,3), filters=512, padding='SAME', \
                              name=layerName, bias_initializer=self.initB, \
                              kernel_initializer=self.initW, use_bias=True, trainable=True )
        if self.saveLayer:  self.layerOut[ layerName ] = x12
        
        layerName = 'activation' + layerIdx
        #x12 = tf.nn.relu( x12, name=layerName )
        x12 = tf.nn.leaky_relu( x12, alpha=leak, name=layerName )
        if self.saveLayer:  self.layerOut[ layerName ] = x12
        
        # Output size 14 x 14 x 512 (H x W x D).
        
#-------------------------------------------------------------------------------
        
        # Input size 14 x 14 x 512 (H x W x D).
        layerIdx = '13'
        layerName = 'conv' + layerIdx
        x13 = tf.layers.conv2d( x12, kernel_size=(3,3), filters=512, padding='SAME', \
                              name=layerName, bias_initializer=self.initB, \
                              kernel_initializer=self.initW, use_bias=True, trainable=True )
        if self.saveLayer:  self.layerOut[ layerName ] = x13
        
        layerName = 'activation' + layerIdx
        #x13 = tf.nn.relu( x13, name=layerName )
        x13 = tf.nn.leaky_relu( x13, alpha=leak, name=layerName )
        if self.saveLayer:  self.layerOut[ layerName ] = x13
        
        # Output size 14 x 14 x 512 (H x W x D).
        
#-------------------------------------------------------------------------------

        # Input size 14 x 14 x 512 (H x W x D).
        layerIdx = '14'
        layerName = 'convtr' + layerIdx
        x14 = tf.layers.conv2d_transpose( x13, kernel_size=(2,2), filters=256, \
                                        padding='SAME', name=layerName, strides=(2,2), \
                                        bias_initializer=self.initB, trainable=True, \
                                        kernel_initializer=self.initW, use_bias=True )
        if self.saveLayer:  self.layerOut[ layerName ] = x14
        
        layerName = 'activation' + layerIdx
        #x14 = tf.nn.relu( x14, name=layerName )
        x14 = tf.nn.leaky_relu( x14, alpha=leak, name=layerName )
        if self.saveLayer:  self.layerOut[ layerName ] = x14

        # Output size 28 x 28 x 256 (H x W x D).
        
        x14 = tf.concat( [x14, x10], axis=3 )
        if self.saveLayer:  self.layerOut[ layerName ] = x14

        # Output size 28 x 28 x 512 (H x W x D).
           
#-------------------------------------------------------------------------------

        # Input size 28 x 28 x 512 (H x W x D).
        layerIdx = '15'
        layerName = 'conv' + layerIdx
        x15 = tf.layers.conv2d( x14, kernel_size=(3,3), filters=256, padding='SAME', \
                              name=layerName, bias_initializer=self.initB, \
                              kernel_initializer=self.initW, use_bias=True, trainable=True )
        if self.saveLayer:  self.layerOut[ layerName ] = x15
        
        layerName = 'activation' + layerIdx
        #x15 = tf.nn.relu( x15, name=layerName )
        x15 = tf.nn.leaky_relu( x15, alpha=leak, name=layerName )
        if self.saveLayer:  self.layerOut[ layerName ] = x15
        
        # Output size 28 x 28 x 256 (H x W x D).
        
#-------------------------------------------------------------------------------

        # Input size 28 x 28 x 256 (H x W x D).
        layerIdx = '16'
        layerName = 'conv' + layerIdx
        x16 = tf.layers.conv2d( x15, kernel_size=(3,3), filters=256, padding='SAME', \
                              name=layerName, bias_initializer=self.initB, \
                              kernel_initializer=self.initW, use_bias=True, trainable=True )
        if self.saveLayer:  self.layerOut[ layerName ] = x16
        
        layerName = 'activation' + layerIdx
        #x16 = tf.nn.relu( x16, name=layerName )
        x16 = tf.nn.leaky_relu( x16, alpha=leak, name=layerName )
        if self.saveLayer:  self.layerOut[ layerName ] = x16
        
        # Output size 28 x 28 x 256 (H x W x D).
        
#-------------------------------------------------------------------------------

        # Input size 28 x 28 x 256 (H x W x D).
        layerIdx = '17'
        layerName = 'convtr' + layerIdx
        x17 = tf.layers.conv2d_transpose( x16, kernel_size=(2,2), filters=128, \
                                        padding='SAME', name=layerName, strides=(2,2), \
                                        bias_initializer=self.initB, trainable=True, \
                                        kernel_initializer=self.initW, use_bias=True )
        if self.saveLayer:  self.layerOut[ layerName ] = x17
        
        layerName = 'activation' + layerIdx
        #x17 = tf.nn.relu( x17, name=layerName )
        x17 = tf.nn.leaky_relu( x17, alpha=leak, name=layerName )
        if self.saveLayer:  self.layerOut[ layerName ] = x17

        # Output size 56 x 56 x 128 (H x W x D).
        
        x17 = tf.concat( [x17, x7], axis=3 )
        if self.saveLayer:  self.layerOut[ layerName ] = x17

        # Output size 56 x 56 x 256 (H x W x D).
           
#-------------------------------------------------------------------------------

        # Input size 56 x 56 x 256 (H x W x D).
        layerIdx = '18'
        layerName = 'conv' + layerIdx
        x18 = tf.layers.conv2d( x17, kernel_size=(3,3), filters=128, padding='SAME', \
                              name=layerName, bias_initializer=self.initB, \
                              kernel_initializer=self.initW, use_bias=True, trainable=True )
        if self.saveLayer:  self.layerOut[ layerName ] = x18
        
        layerName = 'activation' + layerIdx
        #x18 = tf.nn.relu( x18, name=layerName )
        x18 = tf.nn.leaky_relu( x18, alpha=leak, name=layerName )
        if self.saveLayer:  self.layerOut[ layerName ] = x18
        
        # Output size 56 x 56 x 128 (H x W x D).
        
#-------------------------------------------------------------------------------

        # Input size 56 x 56 x 128 (H x W x D).
        layerIdx = '19'
        layerName = 'conv' + layerIdx
        x19 = tf.layers.conv2d( x18, kernel_size=(3,3), filters=128, padding='SAME', \
                              name=layerName, bias_initializer=self.initB, \
                              kernel_initializer=self.initW, use_bias=True, trainable=True )
        if self.saveLayer:  self.layerOut[ layerName ] = x19
        
        layerName = 'activation' + layerIdx
        #x19 = tf.nn.relu( x19, name=layerName )
        x19 = tf.nn.leaky_relu( x19, alpha=leak, name=layerName )
        if self.saveLayer:  self.layerOut[ layerName ] = x19
        
        # Output size 56 x 56 x 128 (H x W x D).
        
#-------------------------------------------------------------------------------

        # Input size 56 x 56 x 128 (H x W x D).
        layerIdx = '20'
        layerName = 'convtr' + layerIdx
        x20 = tf.layers.conv2d_transpose( x19, kernel_size=(2,2), filters=64, \
                                        padding='SAME', name=layerName, strides=(2,2), \
                                        bias_initializer=self.initB, trainable=True, \
                                        kernel_initializer=self.initW, use_bias=True )
        if self.saveLayer:  self.layerOut[ layerName ] = x20
        
        layerName = 'activation' + layerIdx
        #x20 = tf.nn.relu( x20, name=layerName )
        x20 = tf.nn.leaky_relu( x20, alpha=leak, name=layerName )
        if self.saveLayer:  self.layerOut[ layerName ] = x20

        # Output size 112 x 112 x 64 (H x W x D).
        
        x20 = tf.concat( [x20, x4], axis=3 )
        if self.saveLayer:  self.layerOut[ layerName ] = x20

        # Output size 112 x 112 x 128 (H x W x D).
           
#-------------------------------------------------------------------------------

        # Input size 112 x 112 x 128 (H x W x D).
        layerIdx = '21'
        layerName = 'conv' + layerIdx
        x21 = tf.layers.conv2d( x20, kernel_size=(3,3), filters=64, padding='SAME', \
                              name=layerName, bias_initializer=self.initB, \
                              kernel_initializer=self.initW, use_bias=True, trainable=True )
        if self.saveLayer:  self.layerOut[ layerName ] = x21
        
        layerName = 'activation' + layerIdx
        #x21 = tf.nn.relu( x21, name=layerName )
        x21 = tf.nn.leaky_relu( x21, alpha=leak, name=layerName )
        if self.saveLayer:  self.layerOut[ layerName ] = x21
        
        # Output size 112 x 112 x 64 (H x W x D).
        
#-------------------------------------------------------------------------------

        # Input size 112 x 112 x 64 (H x W x D).
        layerIdx = '22'
        layerName = 'conv' + layerIdx
        x22 = tf.layers.conv2d( x21, kernel_size=(3,3), filters=64, padding='SAME', \
                              name=layerName, bias_initializer=self.initB, \
                              kernel_initializer=self.initW, use_bias=True, trainable=True )
        if self.saveLayer:  self.layerOut[ layerName ] = x22
        
        layerName = 'activation' + layerIdx
        #x22 = tf.nn.relu( x22, name=layerName )
        x22 = tf.nn.leaky_relu( x22, alpha=leak, name=layerName )
        if self.saveLayer:  self.layerOut[ layerName ] = x22
        
        # Output size 112 x 112 x 64 (H x W x D).
        
#-------------------------------------------------------------------------------

        # Input size 112 x 112 x 64 (H x W x D).
        layerIdx = '23'
        layerName = 'convtr' + layerIdx
        x23 = tf.layers.conv2d_transpose( x22, kernel_size=(2,2), filters=32, \
                                        padding='SAME', name=layerName, strides=(2,2), \
                                        bias_initializer=self.initB, trainable=True, \
                                        kernel_initializer=self.initW, use_bias=True )
        if self.saveLayer:  self.layerOut[ layerName ] = x23
        
        layerName = 'activation' + layerIdx
        #x23 = tf.nn.relu( x23, name=layerName )
        x23 = tf.nn.leaky_relu( x23, alpha=leak, name=layerName )
        if self.saveLayer:  self.layerOut[ layerName ] = x23

        # Output size 224 x 224 x 32 (H x W x D).
        
        x23 = tf.concat( [x23, x1], axis=3 )
        if self.saveLayer:  self.layerOut[ layerName ] = x23

        # Output size 224 x 224 x 64 (H x W x D).
           
#-------------------------------------------------------------------------------

        # Input size 224 x 224 x 64 (H x W x D).
        layerIdx = '24'
        layerName = 'conv' + layerIdx
        x24 = tf.layers.conv2d( x23, kernel_size=(3,3), filters=32, padding='SAME', \
                              name=layerName, bias_initializer=self.initB, \
                              kernel_initializer=self.initW, use_bias=True, trainable=True )
        if self.saveLayer:  self.layerOut[ layerName ] = x24
        
        layerName = 'activation' + layerIdx
        #x24 = tf.nn.relu( x24, name=layerName )
        x24 = tf.nn.leaky_relu( x24, alpha=leak, name=layerName )
        if self.saveLayer:  self.layerOut[ layerName ] = x24
        
        # Output size 224 x 224 x 32 (H x W x D).
        
#-------------------------------------------------------------------------------

        # Input size 224 x 224 x 32 (H x W x D).
        layerIdx = '25'
        layerName = 'conv' + layerIdx
        x25 = tf.layers.conv2d( x24, kernel_size=(3,3), filters=32, padding='SAME', \
                              name=layerName, bias_initializer=self.initB, \
                              kernel_initializer=self.initW, use_bias=True, trainable=True )
        if self.saveLayer:  self.layerOut[ layerName ] = x25
        
        layerName = 'activation' + layerIdx
        #x25 = tf.nn.relu( x25, name=layerName )
        x25 = tf.nn.leaky_relu( x25, alpha=leak, name=layerName )
        if self.saveLayer:  self.layerOut[ layerName ] = x25
        
        # Output size 224 x 224 x 32 (H x W x D).
        
#-------------------------------------------------------------------------------

        # Input size 224 x 224 x 32 (H x W x D).
        layerIdx = '26'
        layerName = 'conv' + layerIdx
        
        # Since this is basically creating a one hot vector for every pixel, so for 
        # the pixels that are just background, and not part of any object, there 
        # should be a channel as well. Hence a channel is added for the all black 
        # pixels as well. So for 10 classes the number of channels for this segmentLabel
        # array will be 11.

        x26 = tf.layers.conv2d( x25, kernel_size=(1,1), filters=nClasses+1, padding='SAME', \
                              name=layerName, bias_initializer=self.initB, \
                              kernel_initializer=self.initW, use_bias=True, trainable=True )
        if self.saveLayer:  self.layerOut[ layerName ] = x26
        
        # Output size 224 x 224 x nClasses+1 (H x W x D).
        
        return x26
    
#===============================================================================

    def loss(self, logits, labels, weights):
        '''
        Defines the prediction loss. This is a pixelwise softmax over the entire 
        feature map.
        '''
        labels = tf.convert_to_tensor( labels )
        labels = tf.cast( labels, dtype=tf.float32 )

        # The softmax_cross_entropy_with_logits_v2 takes in raw logits as input.
        # It converts the raw logits into softmax internally before calculating 
        # the loss. Hence, the softmax should not be calculated separately. 
        # By default the last dimension is considered as the class dimension along
        # which the cross entropy is calculated.
        weights = tf.convert_to_tensor( weights, dtype=tf.float32 )
        
        lossTensor = tf.losses.softmax_cross_entropy( logits=logits, \
                                            onehot_labels=labels, weights=weights )

        #lossTensor = tf.compat.v1.losses.softmax_cross_entropy( logits=logits, onehot_labels=labels )
        #lossTensor = tf.nn.softmax_cross_entropy_with_logits_v2( logits=logits, labels=labels )
        
        # Return the average loss over this batch.
        #return tf.reduce_sum( lossTensor )
        return tf.reduce_mean( lossTensor )
        
#===============================================================================

    def train( self, trainDir=None, validDir=None ):
        '''
        Trains the model.
        '''
        if trainDir is None or validDir is None:
            print( '\nERROR: one or more input arguments missing ' \
                   'in train. Aborting.\n' )
            sys.exit() 
        
        # SET INPUTS AND LABELS.
        
        # Batch size will be set during runtime as the last batch may not be of
        # the same size of the other batches (especially the last batch).
        x = tf.placeholder( dtype=tf.float32, name='xPlaceholder', \
                            shape=[ None, inImgH, inImgW, 3 ] )
        
        # Labels are one hot vectors.
        # Since this is basically creating a one hot vector for every pixel, so for 
        # the pixels that are just background, and not part of any object, there 
        # should be a channel as well. Hence a channel is added for the all black 
        # pixels as well. So for 10 classes the number of channels for this segmentLabel
        # array will be 11.
        y = tf.placeholder( dtype=tf.float32, name='yPlaceholder', \
                            shape=[ None, inImgH, inImgW, nClasses+1 ] )
        
        # Placeholder for the weights of the segmentation map.
        w = tf.placeholder( dtype=tf.float32, name='wPlaceholder', \
                            shape=[ None, inImgH, inImgW ] )
        
#-------------------------------------------------------------------------------

        # EVALUATE MODEL OUTPUT.
        
        with tf.variable_scope( modelName, reuse=tf.AUTO_REUSE ):
            # AUTO_REUSE flag is used so that no error is there when the same 
            # model parameters are used to check multiple images in sequence.
            predLogits = self.model(x)     # Model prediction logits.
            predProb = tf.nn.softmax( predLogits, axis=-1 )    # Predicted probabilities.
            
            # List of model variables.
            listOfModelVars = []
            for v in tf.global_variables():     listOfModelVars.append( v )

            #for v in listOfModelVars:
                #print( 'Model: {}, Variable: {}'.format( modelName, v ) )
            
#-------------------------------------------------------------------------------
        
        # CALCULATE LOSS.
        
        loss = self.loss( logits=predLogits, labels=y, weights=w )
        
        # DEFINE OPTIMIZER AND PERFORM BACKPROP.
        optimizer = self.optimizer
        
        # While executing an operation (such as trainStep), only the subgraph 
        # components relevant to trainStep will be executed. The 
        # update_moving_averages operation (for the batch normalization layers) 
        # is not a parent of trainStep in the computational graph, so it will 
        # never update the moving averages by default. To get around this, 
        # we have to explicitly tell the graph in the following manner.
        update_ops = tf.get_collection( tf.GraphKeys.UPDATE_OPS )
        with tf.control_dependencies( update_ops ):
            trainStep = optimizer.minimize( loss )

        # List of optimizer variables.
        listOfOptimizerVars = list( set( tf.global_variables() ) - set( listOfModelVars ) )

        #for v in listOfOptimizerVars:
            #print( 'Optimizer: {}, Variable: {}'.format( self.optimizerName, v ) )

#-------------------------------------------------------------------------------
        
        # CREATE A LISTS TO HOLD ACCURACY AND LOSS VALUES.
        
        # This list will have strings for each epoch. Each of these strings 
        # will be like the following:
        # "epoch, learningRate, trainLoss, trainAcc (%), validLoss, validAcc (%)"
        statistics = []
        
        # Format of the statistics values.
        statisticsFormat = 'epoch, learningRate, batchSize, trainLoss, trainAcc (%), ' \
                           'validLoss, validAcc (%), epochProcessTime'
                        
#-------------------------------------------------------------------------------
        
        # START SESSION.
        
        sess = tf.Session()
        print( '\nStarting session. Optimizer: {}, Learning Rate: {}, ' \
               'Batch Size: {}'.format( self.optimizerName, learningRate, batchSize ) )
        
        self.isTraining = True    # Enabling the training flag. 
        
        # Define model saver.
        # Finding latest checkpoint and the latest completed epoch.
        jsonFilePath, ckptPath, latestEpoch = findLatestCkpt( ckptDirPath, \
                                                              training=self.isTraining )
        startEpoch = latestEpoch + 1    # Start from the next epoch.

#-------------------------------------------------------------------------------
        
        # LOAD PARAMETERS FROM CHECKPOINTS.
        
        if ckptPath != None:    # Only when some checkpoint is found.
            with open( jsonFilePath, 'r' ) as infoFile:
                infoDict = json.load( infoFile )
                
            if infoDict['learningRate'] == learningRate and infoDict['batchSize'] == batchSize:
                # Since all old variables will be loaded here, so we do not need
                # to initialize any other variables.
                
                # Now define a new saver with all the variables.
                saver = tf.train.Saver( listOfModelVars + listOfOptimizerVars )
                saver.max_to_keep = nSavedCkpt    # Save upto 5 checkpoints.
                saver.restore( sess, ckptPath )

                print( '\nReloaded ALL variables from checkpoint: {}\n'.format( \
                                                            ckptPath ) )

#-------------------------------------------------------------------------------

            else:
                # else load only weight and biases and skip optimizer 
                # variables. Otherwise there will be errors.
                # So define the saver with the list of only model variables.
                saver = tf.train.Saver( listOfModelVars )
                saver.restore( sess, ckptPath )
                # Since only the model variables will be loaded here, so we
                # have to initialize other variables (optimizer variables)
                # separately.
                sess.run( tf.variables_initializer( listOfOptimizerVars ) )

                # The previous saver will only save the listOfModelVars
                # as it is defined using only those variables (as the 
                # checkpoint can only give us those values as valid for
                # restoration). But now since we have all the varibles 
                # loaded and all the new ones initialized, so we redefine 
                # the saver to include all the variables. So that while 
                # saving in the end of the epoch, it can save all the 
                # variables (and not just the listOfModelVars).
                saver = tf.train.Saver( listOfModelVars + listOfOptimizerVars )
                saver.max_to_keep = nSavedCkpt    # Save upto 5 checkpoints.

                # Load mean and std values.
                mean, std = infoDict[ 'mean' ], infoDict[ 'std' ]
                
                print( '\nCurrent parameters:\nlearningRate: {}, batchSize: {}' \
                       '\nPrevious parameters (inside checkpoint {}):\nlearningRate: ' \
                       '{}, batchSize: {}\nThey are different.\nSo reloaded only ' \
                       'MODEL variables from checkpoint: {}\nAnd initialized ' \
                       'other variables.\n'.format( learningRate, batchSize, ckptPath, \
                       infoDict[ 'learningRate' ], infoDict[ 'batchSize' ], ckptPath ) )
                
            # Reloading accuracy and loss statistics, mean and std from checkpoint.
            statistics = infoDict[ 'statistics' ]
            mean = np.array( infoDict[ 'mean' ] )
            std = np.array( infoDict[ 'std' ] )
            maxValidAcc = infoDict[ 'maxValidAcc' ]
            minValidLoss = infoDict[ 'minValidLoss' ]
            
            # If the batchsize changes, then the minValidLoss should also be 
            # scaled according to that.
            if batchSize != infoDict[ 'batchSize' ]:
                minValidLoss = infoDict['minValidLoss'] * batchSize / infoDict['batchSize']

#-------------------------------------------------------------------------------

        else:
            # When there are no valid checkpoints initialize the saver to 
            # save all parameters.
            saver = tf.train.Saver( listOfModelVars + listOfOptimizerVars )
            saver.max_to_keep = nSavedCkpt    # Save upto 5 checkpoints.

            sess.run( tf.variables_initializer( listOfModelVars + listOfOptimizerVars ) )
           
            # Calculate mean and std.
            if recordedMean is None or recordedStd is None:
                mean, std = datasetMeanStd( trainDir )
                print( '\nMean of {}: {}'.format( trainDir, mean ) )
                print( '\nStd of {}: {}'.format( trainDir, std ) )
            else:
                mean, std = recordedMean, recordedStd

            maxValidAcc = 0.0
            minValidLoss = np.inf

#-------------------------------------------------------------------------------
        
        # TRAINING AND VALIDATION.
        
        print( '\nStarted Training...\n' )
        
        for epoch in range( startEpoch, nEpochs+1 ):
            # epoch will be numbered from 1 to 150 if there are 150 epochs.
            # Is is not numbered from 0 to 149.

            epochProcessTime = time.time()
            
            # TRAINING PHASE.
            
            self.isTraining = True    # Enabling training flag at the start of 
            # the training phase of each epoch as it will be disabled in the 
            # corresponding validaton phase. 

            # This list contains the filepaths for all the images in trainDir.
            # If there are unwanted files like '.thumbs' etc. then those are 
            # filtered as well.
            listOfRemainingTrainImg = [ i for i in os.listdir( \
                                        os.path.join( trainDir, 'images' ) ) \
                                                if i[-3:] == 'bmp' ]

            nTrainImgs = len( listOfRemainingTrainImg )
            nTrainBatches = int( np.ceil( nTrainImgs / batchSize ) )
            
            # Interval at which the status will be printed in the terminal.
            printInterval = 1 if nTrainBatches < 10 else int( nTrainBatches / 10 )
            
            trainAcc = 0.0
            trainLoss = 0.0
            trainBatchIdx = 0    # Counts the number of batch processed.
            
            # Storing information of current epoch for recording later in statistics.
            # This is recorded as a string for better visibility in the json file.
            currentStatistics = '{}, {}, {}, '.format( epoch, learningRate, batchSize )
            
#-------------------------------------------------------------------------------
            
            # Scan entire training dataset.
            while len( listOfRemainingTrainImg ) > 0:

                trainBatchProcessTime = time.time()

                # Creating batches. Also listOfRemainingTrainImg is updated.
                trainImgBatch, trainLabelBatch, trainWeightBatch, listOfRemainingTrainImg, _ = \
                    createBatchForSegmentation( trainDir, listOfRemainingTrainImg, \
                                                  batchSize, shuffle=True )
                
                feedDict = { x: trainImgBatch, y: trainLabelBatch, w: trainWeightBatch }
                if self.saveLayer:    # Evaluate layer outputs if this flag is True.
                    trainLayerOut = sess.run( self.layerOut, feed_dict=feedDict )

                trainPredLogits = sess.run( predLogits, feed_dict=feedDict )
                trainBatchLoss = sess.run( loss, feed_dict=feedDict )
                sess.run( trainStep, feed_dict=feedDict )
                
                trainLoss += ( trainBatchLoss / nTrainBatches )
                
                # The trainPredLogits is an array of logits. It needs to be
                # converted to sigmoid to get probability and then we need
                # to extract the max index to get the labels.
                trainPredProb = sess.run( predProb, feed_dict=feedDict )
                
                #print( sess.run(tf.reduce_sum(trainPredProb)) )
                #print( sess.run(tf.reduce_sum(trainPredLogits)) )
                #print( sess.run(tf.reduce_sum(tf.nn.softmax(trainPredLogits, axis=-1))) )
                #print( sess.run(tf.reduce_sum(trainLabelBatch)) )

                # If the probability is more than the threshold, the 
                # corresponding label element is considered as 1 else 0.
                trainPredLabel = np.asarray( trainPredProb > threshProb, dtype=np.int32 )
                
#-------------------------------------------------------------------------------                
                
                matches = np.array( trainPredLabel == trainLabelBatch, dtype=np.int32 )
                
                # This will be an array of batchSize x 5 (as there are 5 classes).
                # A completely correct prediction will have a True match 
                # for all of these 5 elements. The sum of this matches array is
                # calculated along the channel axis. If that results in 5 then it 
                # means that a perfect match has happened.
                matches = matches[ :, :, :, :-1 ]   # Ignoring the background channel.
                matches = np.sum( matches, axis=-1 )
                perfectMatch = np.asarray( np.ones( matches.shape ) * (nClasses), dtype=np.int32 )
                matches1 = np.array( matches == perfectMatch, dtype=np.int32 )
                
                # The sum will include all the pixels. Hence taking the average over 
                # all the pixels.
                trainAcc += ( 100*np.sum( matches1 ) ) / ( nTrainImgs*inImgH*inImgW )
                
                trainBatchIdx += 1
                trainBatchProcessTime = time.time() - trainBatchProcessTime

                # Print the status on the terminal every 10 batches.
                if trainBatchIdx % printInterval == 0:
                    print( 'Epoch: {}/{},\tBatch: {}/{},\tBatch loss: ' \
                           '{:0.6f},\tProcess time for {} batch: {}'.format( epoch, \
                           nEpochs, trainBatchIdx, nTrainBatches, trainBatchLoss, \
                           printInterval, prettyTime( trainBatchProcessTime*printInterval ) ) )
            
            ## Recording training loss and accuracy in current statistics string.
            #currentStatistics += '{}, '.format( trainLoss )

            # Recording training loss and accuracy in current statistics string.
            currentStatistics += '{}, {}, '.format( trainLoss, trainAcc )
            
#-------------------------------------------------------------------------------
            
            # VALIDATION PHASE.
            
            self.isTraining = False    # Disabling the training flag.
            
            # This list contains the filepaths for all the images in validDir.
            # If there are unwanted files like '.thumbs' etc. then those are 
            # filtered as well.
            listOfRemainingValidImg = [ i for i in os.listdir( \
                                        os.path.join( validDir, 'images' ) ) \
                                                if i[-3:] == 'bmp' ]
            
            nValidImgs = len( listOfRemainingValidImg )
            nValidBatches = int( np.ceil( nValidImgs / batchSize ) )
            
            # Interval at which the status will be printed in the terminal.
            printInterval = 1 if nValidBatches < 3 else int( nValidBatches / 3 )

            validAcc = 0.0
            validLoss = 0.0
            validBatchIdx = 0    # Counts the number of batch processed.
            
            print( '\n\nValidation phase for epoch {}.\n'.format( epoch ) )
            
#-------------------------------------------------------------------------------
            
            # Scan entire validation dataset.
            while len( listOfRemainingValidImg ) > 0:
                
                validBatchProcessTime = time.time()

                # Creating batches. Also listOfRemainingValidImg is updated.
                # The shuffle is off for validation and the mean and std are
                # the same as calculated on the training set.
                validImgBatch, validLabelBatch, validWeigthBatch, listOfRemainingValidImg, _ = \
                    createBatchForSegmentation( validDir, listOfRemainingValidImg, \
                                                  batchSize, shuffle=False )
                
                feedDict = { x: validImgBatch, y: validLabelBatch, w: validWeigthBatch }
                if self.saveLayer:    # Evaluate layer outputs if this flag is True.
                    validLayerOut = sess.run( self.layerOut, feed_dict=feedDict )
                
                validPredLogits = sess.run( predLogits, feed_dict=feedDict )
                validBatchLoss = sess.run( loss, feed_dict=feedDict )
                
                validLoss += ( validBatchLoss / nValidBatches )
                                    
                # The validPredLogits is an array of logits. It needs to be 
                # converted to sigmoid to get probability and then we need
                # to extract the max index to get the labels.
                validPredProb = sess.run( predProb, feed_dict=feedDict )
                
                # If the probability is more than the threshold, the 
                # corresponding label element is considered as 1 else 0.
                validPredLabel = np.asarray( validPredProb > threshProb, dtype=np.int32 )
                
#------------------------------------------------------------------------------- 
                
                matches = np.array( validPredLabel == validLabelBatch, dtype=np.int32 )
                
                # This will be an array of batchSize x 5 (as there are 5 classes).
                # A completely correct prediction will have a True match 
                # for all of these 5 elements. The sum of this matches array is
                # calculated along the channel axis. If that results in 5 then it 
                # means that a perfect match has happened.
                matches = matches[ :, :, :, :-1 ]   # Ignoring the background channel.
                matches = np.sum( matches, axis=-1 )
                perfectMatch = np.asarray( np.ones( matches.shape ) * (nClasses), dtype=np.int32 )
                matches1 = np.array( matches == perfectMatch, dtype=np.int32 )
                
                # The sum will include all the pixels. Hence taking the average over 
                # all the pixels.
                validAcc += ( 100*np.sum( matches1 ) ) / ( nValidImgs*inImgH*inImgW )
                
                validBatchIdx += 1
                validBatchProcessTime = time.time() - validBatchProcessTime

                # Print the status on the terminal every 10 batches.
                if validBatchIdx % printInterval == 0:     
                    print( 'Epoch: {}/{},\tBatch: {}/{},\tBatch loss: ' \
                           '{:0.6f},\tProcess time for {} batch: {}'.format( epoch, \
                           nEpochs, validBatchIdx, nValidBatches, validBatchLoss, \
                           printInterval, prettyTime( validBatchProcessTime*printInterval ) ) )

            ## Recording validation accuracy in current statistics string.
            #currentStatistics += '{}, '.format( validLoss )

            # Recording validation accuracy in current statistics string.
            currentStatistics += '{}, {}, '.format( validLoss, validAcc )

#-------------------------------------------------------------------------------            

            # STATUS UPDATE.
            
            epochProcessTime = time.time() - epochProcessTime
            
            # Recording epoch processing time in current statistics string.
            currentStatistics += '{}'.format( prettyTime( epochProcessTime ) )
            
            # Noting accuracy after the end of all batches.
            statistics.append( currentStatistics )
            
            ## Printing current epoch.
            #print( '\nEpoch {} done. Epoch time: {}, Train ' \
                   #'loss: {:0.6f}, Valid loss: {:0.6f}\n'.format( epoch, \
                   #prettyTime( epochProcessTime ), trainLoss, validLoss ) )

            # Printing current epoch.
            print( '\nEpoch {} done. Epoch time: {}, Train ' \
                   'loss: {:0.6f}, Train Accuracy: {:0.3f} %, Valid loss: {:0.6f}, ' \
                   'Valid Accuracy: {:0.3f} %\n'.format( epoch, \
                   prettyTime( epochProcessTime ), trainLoss, trainAcc, validLoss, \
                   validAcc ) )

            # Saving the variables at some intervals, only if there is 
            # improvement in validation accuracy.

            #if epoch % modelSaveInterval == 0 and validLoss < minValidLoss:
                #ckptSavePath = os.path.join( ckptDirPath, savedCkptName )
                #saver.save( sess, save_path=ckptSavePath, global_step=epoch )
                
            if (epoch % modelSaveInterval == 0 and \
               ( (validLoss < minValidLoss) or (validAcc > maxValidAcc) ) ) or epoch < 3:
                ckptSavePath = os.path.join( ckptDirPath, savedCkptName )
                saver.save( sess, save_path=ckptSavePath, global_step=epoch )

                maxValidAcc = validAcc      # Updating the maxValidAcc.
                minValidLoss = validLoss    # Updating the minValidLoss.
                
                # Saving the important info like learning rate, batch size,
                # and training error for the current epoch in a json file.
                # Converting the mean and std into lists before storing as
                # json cannot store numpy arrays. And also saving the training
                # and validation loss and accuracy statistics.
                jsonInfoFilePath = ckptSavePath + '-' + str( epoch ) + '.json'
                with open( jsonInfoFilePath, 'w' ) as infoFile:
                    infoDict = { 'epoch': epoch, 'batchSize': batchSize, \
                                 'learningRate': learningRate, \
                                 'mean': list( mean ), 'std': list( std ), \
                                 'maxValidAcc': maxValidAcc, \
                                 'minValidLoss': minValidLoss, \
                                 'statisticsFormat': statisticsFormat, \
                                 'statistics': statistics }
                    
                    json.dump( infoDict, infoFile, sort_keys=False, \
                               indent=4, separators=(',', ': ') )
                
                print( 'Checkpoint saved.\n' )

            ## Updating the maxValidAcc value.
            #elif validLoss < minValidLoss:
                #minValidLoss = validLoss
            
            # Updating the maxValidAcc value.
            elif (validLoss < minValidLoss) or (validAcc > maxValidAcc):
                maxValidAcc = validAcc
                minValidLoss = validLoss

#-------------------------------------------------------------------------------
        
        self.isTraining = False   # Indicates the end of training.
        print( '\nTraining completed with {} epochs.'.format( nEpochs ) )

        sess.close()        # Closing the session.
        tf.reset_default_graph()    # Reset default graph, else it will be slow if rerun in loop.

#===============================================================================

    def test( self, testDir=None ):
        '''
        Tests the model.
        '''
        if testDir is None:
            print( '\nERROR: one or more input arguments missing ' \
                   'in test. Aborting.\n' )
            sys.exit() 

        self.isTraining = False    # Disabling the training flag.
        
        # SET INPUTS AND LABELS.
        
        # Batch size will be set during runtime as the last batch may not be of
        # the same size of the other batches (especially the last batch).
        x = tf.placeholder( dtype=tf.float32, name='xPlaceholder', \
                            shape=[ None, inImgH, inImgW, 3 ] )
        
        # Labels are one hot vectors.
        # Since this is basically creating a one hot vector for every pixel, so for 
        # the pixels that are just background, and not part of any object, there 
        # should be a channel as well. Hence a channel is added for the all black 
        # pixels as well. So for 10 classes the number of channels for this segmentLabel
        # array will be 11.
        y = tf.placeholder( dtype=tf.float32, name='yPlaceholder', \
                            shape=[ None, inImgH, inImgW, nClasses+1 ] )

        # Placeholder for the weights of the segmentation map.
        w = tf.placeholder( dtype=tf.float32, name='wPlaceholder', \
                            shape=[ None, inImgH, inImgW ] )
        
#-------------------------------------------------------------------------------

        # EVALUATE MODEL OUTPUT.
        
        with tf.variable_scope( modelName, reuse=tf.AUTO_REUSE ):
            # AUTO_REUSE flag is used so that no error is there when the same 
            # model parameters are used to check multiple images in sequence.
            predLogits = self.model(x)     # Model prediction logits.
            predProb = tf.nn.softmax( predLogits, axis=-1 )    # Predicted probabilities.
            
            # List of model variables.
            listOfModelVars = []
            for v in tf.global_variables():
                # Include only those variables whose name has this model's name in it.
                # Also, there is no need to include the optimizer variables as there 
                # is no training going on.
                if v.name.find( modelName ) >= 0:
                    listOfModelVars.append( v )
                    #print( 'Model: {}, Variable: {}'.format( modelName, v ) )

#-------------------------------------------------------------------------------

        # CALCULATE LOSS.
        
        loss = self.loss( logits=predLogits, labels=y, weights=w )

#-------------------------------------------------------------------------------
        
        # START SESSION.
        
        sess = tf.Session()
        
        # Define model saver.
        # Finding latest checkpoint and the latest completed epoch.
        jsonFilePath, ckptPath, latestEpoch = findLatestCkpt( ckptDirPath, \
                                                              training=self.isTraining )

        if ckptPath != None:    # Only when some checkpoint is found.
            saver = tf.train.Saver( listOfModelVars )
            saver.restore( sess, ckptPath )            
            print( '\nReloaded ALL variables from checkpoint: {}'.format( \
                                                        ckptPath ) )
        else:
            # When there are no valid checkpoints.
            print( '\nNo valid checkpoints found. Aborting.\n' )
            return

        with open( jsonFilePath, 'r' ) as infoFile:
            infoDict = json.load( infoFile )

        # Reloading mean and std from checkpoint.
        mean = np.array( infoDict[ 'mean' ] )
        std = np.array( infoDict[ 'std' ] )

#-------------------------------------------------------------------------------
                    
        print( '\nStarted Testing...\n' )
            
        # TESTING PHASE.

        testingTime = time.time()

        # This list contains the filepaths for all the images in validDir.
        # If there are unwanted files like '.thumbs' etc. then those are 
        # filtered as well.
        listOfRemainingTestImg = [ i for i in os.listdir( \
                                  os.path.join( testDir, 'images' ) ) \
                                            if i[-3:] == 'bmp' ]
        
        nTestImgs = len( listOfRemainingTestImg )
        nTestBatches = int( np.ceil( len( listOfRemainingTestImg ) / batchSize ) )
        
        # Interval at which the status will be printed in the terminal.
        printInterval = 1 if nTestBatches < 3 else int( nTestBatches / 3 )
        
        testAcc = 0.0
        testLoss = 0.0
        testBatchIdx = 0    # Counts the number of batches processed.

#-------------------------------------------------------------------------------
        
        # Scan entire validation dataset.
        while len( listOfRemainingTestImg ) > 0:
        
            testBatchProcessTime = time.time()

            # Creating batches. Also listOfRemainingTestImg is updated.
            # The shuffle is off for validation and the mean and std are
            # the same as calculated on the training set.
            testImgBatch, testLabelBatch, testWeightBatch, listOfRemainingTestImg, \
                listOfSelectedTestImg = \
                createBatchForSegmentation( testDir, listOfRemainingTestImg, \
                                              batchSize, shuffle=False )
            
            feedDict = { x: testImgBatch, y: testLabelBatch, w: testWeightBatch }
            if self.saveLayer:    # Evaluate layer outputs if this flag is True.
                testLayerOut = sess.run( self.layerOut, feed_dict=feedDict )
            
            testPredLogits = sess.run( predLogits, feed_dict=feedDict )
            testBatchLoss = sess.run( loss, feed_dict=feedDict )
                
            testLoss += ( testBatchLoss / nTestBatches )
                                                            
            # The testPredLogits is an array of logits. It needs to be 
            # converted to sigmoid to get probability and then we need
            # to extract the max index to get the labels.
            testPredProb = sess.run( predProb, feed_dict=feedDict )
            
            # If the probability is more than the threshold, the 
            # corresponding label element is considered as 1 else 0.
            testPredLabel = np.asarray( testPredProb > threshProb, dtype=np.int32 )
            
#------------------------------------------------------------------------------- 

            matches = np.array( testPredLabel == testLabelBatch, dtype=np.int32 )
            
            # This will be an array of batchSize x 5 (as there are 5 classes).
            # A completely correct prediction will have a True match 
            # for all of these 5 elements. The sum of this matches array is
            # calculated along the channel axis. If that results in 5 then it 
            # means that a perfect match has happened.
            matches = matches[ :, :, :, :-1 ]   # Ignoring the background channel.
            matches = np.sum( matches, axis=-1 )
            perfectMatch = np.asarray( np.ones( matches.shape ) * (nClasses), dtype=np.int32 )
            matches1 = np.array( matches == perfectMatch, dtype=np.int32 )
            
            # The sum will include all the pixels. Hence taking the average over 
            # all the pixels.
            testAcc += ( 100*np.sum( matches1 ) ) / ( nTestImgs*inImgH*inImgW )
            
            testBatchIdx += 1
            testBatchProcessTime = time.time() - testBatchProcessTime

            # Printing current status of testing.
            # print the status on the terminal every 10 batches.
            if testBatchIdx % printInterval == 0:
                print( 'Batch: {}/{},\tBatch loss: {:0.6f},\tProcess time for {} ' \
                       'batch: {}'.format( testBatchIdx, nTestBatches, testBatchLoss, \
                        printInterval, prettyTime( testBatchProcessTime*printInterval ) ) )

#-------------------------------------------------------------------------------

        testingTime = time.time() - testingTime
        print( '\n\nTesting done. Test Loss: {:0.6f}, Test Accuracy: {:0.3f} %, ' \
               'Testing time: {}'.format( testLoss, testAcc, prettyTime( testingTime ) ) )

        sess.close()        # Closing the session.
        tf.reset_default_graph()    # Reset default graph, else it will be slow if rerun in loop.
        
        return testLoss, testAcc, testingTime

#===============================================================================

    def batchInference( self, imgBatch ):
        '''
        This function evaluates the output of the model on an unknown batch of
        images (4d numpy array) and returns the predicted labels as a batch as well.
        '''
        self.isTraining = False    # Disabling the training flag.
        
        # SET INPUTS.
        
        b, h, w, _ = imgBatch.shape

        # All the images in the batch are already resized to the appropriate shape 
        # when the batch was created, hence no need to resize again.
        x = tf.placeholder( dtype=tf.float32, name='xPlaceholder', \
                            shape=[ None, inImgH, inImgW, 3 ] )
        
#-------------------------------------------------------------------------------

        # EVALUATE MODEL OUTPUT.
        
        with tf.variable_scope( modelName, reuse=tf.AUTO_REUSE ):
            # AUTO_REUSE flag is used so that no error is there when the same 
            # model parameters are used to check multiple images in sequence.
            predLogits = self.model(x)     # Model prediction logits.
            predProb = tf.nn.softmax( predLogits, axis=-1 )    # Predicted probabilities.
            
            # List of model variables.
            listOfModelVars = []
            for v in tf.global_variables():
                # Include only those variables whose name has this model's name in it.
                # Also, there is no need to include the optimizer variables as there 
                # is no training going on.
                if v.name.find( modelName ) >= 0:
                    listOfModelVars.append( v )
                    #print( 'Model: {}, Variable: {}'.format( modelName, v ) )

#-------------------------------------------------------------------------------
        
        # START SESSION.
        
        sess = tf.Session()
        
        # Define model saver.
        # Finding latest checkpoint and the latest completed epoch.
        jsonFilePath, ckptPath, latestEpoch = findLatestCkpt( ckptDirPath, \
                                                              training=self.isTraining )

        if ckptPath != None:    # Only when some checkpoint is found.
            saver = tf.train.Saver( listOfModelVars )
            saver.restore( sess, ckptPath )            
#            print( '\nReloaded ALL variables from checkpoint: {}'.format( \
#                                                        ckptPath ) )
        else:
            # When there are no valid checkpoints.
            print( '\nNo valid checkpoints found. Aborting.\n' )
            return

        with open( jsonFilePath, 'r' ) as infoFile:
            infoDict = json.load( infoFile )
                
        # Reloading mean and std from checkpoint.
        mean = np.array( infoDict[ 'mean' ] )
        std = np.array( infoDict[ 'std' ] )

#-------------------------------------------------------------------------------

#        # Normalizing by mean and std as done in case of training.
#        imgBatch = (imgBatch - mean) / std

        # Converting image to range 0 to 1.
        # The image is explicitly converted to float32 to match the type 
        # specified in the placeholder. If img would have been directly divided
        # by 127.5, then it would result in np.float64.
        imgBatch = np.asarray( imgBatch, dtype=np.float32 ) / 127.5 - 1.0
        
        feedDict = { x: imgBatch }
        if self.saveLayer:    # Evaluate layer outputs if this flag is True.
            inferLayerOut = sess.run( self.layerOut, feed_dict=feedDict )
        else:   inferLayerOut = None

        inferPredLogits = sess.run( predLogits, feed_dict=feedDict )
                                                                                        
        # The testPredLogits is an array of logits. It needs to be 
        # converted to sigmoid to get probability and then we need
        # to extract the max index to get the labels.
        inferPredProb = sess.run( predProb, feed_dict=feedDict )
        
        # If the probability is more than the threshold, the 
        # corresponding label element is considered as 1 else 0.
        inferPredLabel = np.asarray( inferPredProb > threshProb, dtype=np.int32 )

#-------------------------------------------------------------------------------
        
        sess.close()
        tf.reset_default_graph()    # Reset default graph, else it will be slow if rerun in loop.

        return inferLayerOut, inferPredLogits, inferPredProb, inferPredLabel, mean, std
    
#===============================================================================







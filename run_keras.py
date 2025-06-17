'''
**************************************************************************
||                        SiMa.ai CONFIDENTIAL                          ||
||   Unpublished Copyright (c) 2022-2023 SiMa.ai, All Rights Reserved.  ||
**************************************************************************
 NOTICE:  All information contained herein is, and remains the property of
 SiMa.ai. The intellectual and technical concepts contained herein are 
 proprietary to SiMa and may be covered by U.S. and Foreign Patents, 
 patents in process, and are protected by trade secret or copyright law.

 Dissemination of this information or reproduction of this material is 
 strictly forbidden unless prior written permission is obtained from 
 SiMa.ai.  Access to the source code contained herein is hereby forbidden
 to anyone except current SiMa.ai employees, managers or contractors who 
 have executed Confidentiality and Non-disclosure agreements explicitly 
 covering such access.

 The copyright notice above does not evidence any actual or intended 
 publication or disclosure  of  this source code, which includes information
 that is confidential and/or proprietary, and is a trade secret, of SiMa.ai.

 ANY REPRODUCTION, MODIFICATION, DISTRIBUTION, PUBLIC PERFORMANCE, OR PUBLIC
 DISPLAY OF OR THROUGH USE OF THIS SOURCE CODE WITHOUT THE EXPRESS WRITTEN
 CONSENT OF SiMa.ai IS STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE 
 LAWS AND INTERNATIONAL TREATIES. THE RECEIPT OR POSSESSION OF THIS SOURCE
 CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS TO 
 REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE, USE, OR
 SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.                

**************************************************************************
'''


'''
Author: Mark Harvey
'''


'''
Implements a protoype pipeline using the TensorFlow FP32 model
Usage: python run_fp_pipeline.py
'''

# standard modules
import os, sys
import argparse
import cv2
import numpy as np


# TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
from tensorflow.keras.models import load_model


# custom modules
import config as cfg



# configuration
build_dir=cfg.build_dir
DIVIDER=cfg.DIVIDER



def implement(args):
    '''
    Run inferencing of FP pipeline
    ''' 

    assert (os.path.exists(args.model)), f'Did not find checkpoint at {args.model}...exiting'

    # load trained checkpoint
    model = load_model(args.model,compile=False)
    print(f'Loaded model from {args.model}')

    # get input names and shapes
    for inp in model.inputs:
      print('Input Name:',inp.name, ' Shape:',inp.shape, 'Data type:',inp.dtype)
    for outp in model.outputs:
      print('Output Name:',outp.name, ' Shape:',outp.shape, 'Data type:',outp.dtype)


    # list of test images
    image_list = sorted([f for f in os.listdir(f'{args.test_data}') if f.endswith(('.png','.jpg','.jpeg'))])
    assert (len(image_list)>0), f'Did not find any images at {args.test_data}...exiting'
    print(f'Found {len(image_list)} images in {args.test_data}')


    labels=[8,0,7,9,8,4,5,3,4,0,6]
    accuracy=0

    for i,p in enumerate(image_list):
        # read image
        img = cfg.read_image(f'{args.test_data}/{p}')

        # preprocess
        img = cfg.preprocess_image(img)

        # run inference, returns np array of shape (1,10)
        prediction = model.predict(img,verbose=0)

        # post-process
        prediction = np.argmax(prediction,axis=-1)

        # print result
        print(prediction[0], p)
        if prediction[0]==labels[i]:
            accuracy+=1

    print(f'Accuracy: {(accuracy/len(image_list))*100:.2f}%')

    return



def run_main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--test_data', type=str, default='test_data', help='Path to folder containg test samples. Default is test_data.')
    ap.add_argument('-m', '--model',     type=str, default='keras/mnist_640_480.h5', help='Path to Keras model. Default is keras/mnist_640_480.h5.')
    args = ap.parse_args()


    print('\n'+DIVIDER)
    print('TensorFlow version : ',tf.__version__)
    print(sys.version)
    print(DIVIDER)

    implement(args)

if __name__ == '__main__':
    run_main()
    
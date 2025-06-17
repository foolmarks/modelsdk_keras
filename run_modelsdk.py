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
Load, Quantize, Evaluate, Compile
'''

'''
Author: Mark Harvey, SiMa Technologies
'''

# standard modules
import numpy as np
import logging
import os, sys
import argparse
import cv2

# SDK modules
from afe.load.importers.general_importer import ImporterParams, keras_source
from afe.apis.release_v1 import get_model_sdk_version
from afe.apis.loaded_net import load_model
from afe.apis.defines import default_quantization
from afe.apis.error_handling_variables import enable_verbose_error_messages
from afe.core.utils import length_hinted

# TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf



# custom modules
import config as cfg


# configuration
DIVIDER=cfg.DIVIDER
build_dir=cfg.build_dir


def implement(args):

    # Uncomment the following line to enable verbose error messages.
    enable_verbose_error_messages()


    # get filename from full path
    filename = (os.path.splitext(os.path.basename(args.model)))[0]

    # set an output path for saving results
    output_path=f'{args.build_dir}/{filename}'

    assert (os.path.exists(args.model)), f'Did not find checkpoint at {args.model}...exiting'


    '''
    Interrogate Keras model for input names, shapes
    '''
    model = tf.keras.models.load_model(args.model,compile=False)
    input_shape_dict={}
    input_names_list=[]
    for inp in model.inputs:
      print('Name:',inp.name, ' Shape:',inp.shape, 'Data type:',inp.dtype)
      input_shape_dict[inp.name]=(1,*inp.shape[1:])
      input_names_list.append(inp.name)

    '''
    Load the floating-point Keras model
    Refer to online documentation: https://docs.sima.ai/latest/pages/palette/modelsdk/load_model.html#keras-format-models
    '''
    params: ImporterParams = keras_source(model_path=args.model,
                                          shape_dict=input_shape_dict,
                                          layout="NHWC")
    
    loaded_net = load_model(params)
    print(f'Loaded model from {args.model}')



    '''
    Set up calibration data - the calibration samples should be randomly chosen from the training dataset.
    The calibration data must be in NHWC format even if the original Keras model is NCHW
    Each calibration data sample is supplied as a dictionary, key is input name, value is preprocessed calibration data
    The dictionaries are appended to an iterable variable - a list is used in the example below
    '''
    calibration_data=[]
    # make a list of calibration images
    calib_images = [f for f in os.listdir(args.calib_data) if f.endswith(('.png','.jpg','.jpeg'))]
    print(f'Found {len(calib_images)} calibration images')
    for f in (calib_images):
        # open image to np array, converted to RGB, HWC shape order
        image = cfg.read_image(f'{args.calib_data}/{f}')
        # preprocess the image and then append dictionary to list
        preproc_image = cfg.preprocess_image(image)
        calibration_data.append({input_names_list[0]:preproc_image})


    '''
    Quantize with default parameters
    Refer to online docs: https://developer.sima.ai/apps?id=22ef42b1-3652-4cc7-8019-16b86910ed53
    '''
    print('Quantizing...')
    quant_model = loaded_net.quantize(calibration_data=length_hinted(len(calib_images),calibration_data),
                                      quantization_config=default_quantization,
                                      model_name=filename,
                                      log_level=logging.WARN)

    quant_model.save(model_name=filename, output_directory=output_path)
    print (f'Quantized and saved to {output_path}')



    '''
    Evaluate quantized model
    '''
    print('Evaluating quantized model...')
  
    # list test images
    test_images = sorted([f for f in os.listdir(args.test_data) if f.endswith(('.png','.jpg','.jpeg'))])
    assert (len(test_images)>0), f'Did not find any images at {args.test_data}...exiting'
    num_test_images = min(args.num_test_images, len(test_images))
    test_images = test_images[:num_test_images]
    print(f'Using {num_test_images} out of {len(test_images)} test images')
    
    labels=[8,0,7,9,8,4,5,3,4,0,6]
    accuracy=0
    for i,f in enumerate(test_images):
        # open image to np array, converted to RGB, HWC shape order
        image = cfg.read_image(f'{args.test_data}/{f}')
        
        # preprocess
        preproc_image = cfg.preprocess_image(image)

        # dictionary key is name of input that preprocessed sample will be applied to
        test_data={input_names_list[0]: preproc_image }

        # emulate the quantized model
        prediction = quant_model.execute(test_data, fast_mode=True)

        # post-processing - argmax reduction
        prediction = np.argmax(prediction,axis=-1)

        # print result
        print(prediction[0][0], f)
        if prediction[0]==labels[i]:
            accuracy+=1

    print(f'Accuracy: {(accuracy/len(test_images))*100:.2f}%')


    '''
    Compile
    Refer to online docs: https://developer.sima.ai/apps?id=22ef42b1-3652-4cc7-8019-16b86910ed53
    '''
    quant_model.compile(output_path=output_path,
                        batch_size=args.batch_size,
                        log_level=logging.INFO)  

    print(f'Compiled model written to {output_path}')


    return

    


def run_main():
  
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-bd', '--build_dir',       type=str, default='build', help='Path of build folder. Default is build')
    ap.add_argument('-m',  '--model',           type=str, default='./keras/mnist_640_480.h5', help='path to FP32 Keras model')
    ap.add_argument('-b',  '--batch_size',      type=int, default=1, help="requested batch size of compiled model. Default is 1")
    ap.add_argument('-td', '--test_data',       type=str, default='test_data', help='Path of test data folder. Default is test_data')
    ap.add_argument('-ti', '--num_test_images', type=int, default=10, help='Number of test images. Default is 10')
    ap.add_argument('-cd', '--calib_data',      type=str, default='calib_data', help='Path of calibration data folder. Default is calib_data')
    args = ap.parse_args()

    print('\n'+DIVIDER,flush=True)
    print('Model SDK version',get_model_sdk_version())
    print(sys.version,flush=True)
    print(DIVIDER,flush=True)


    implement(args)

  

if __name__ == '__main__':
  run_main()


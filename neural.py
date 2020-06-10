#!/usr/bin/env
# -*- coding: utf-8 -*-
"""
Routines to apply pre-trained neural networks to images

All networks should be stored within a smartimage in ONNX format.
    https://en.wikipedia.org/wiki/Open_Neural_Network_Exchange
    http://onnx.ai
    https://github.com/onnx/models
"""
try:
    import cv2
except ImportError:
    print('ERR: Missing OpenCV')
    print('You can install using:')
    print('   pip install opencv-contrib-python')
    print('Or visit this page if you want to get tricky:')
    print('   https://www.pyimagesearch.com/opencv-tutorials-resources-guides/')
from .imageRepr import *
from .resizing import *


def loadNeuralNetwork(filename):
    """
    load a neural network model from file

    supports .t7 torch format
    TODO: supports onnx xml exchange format
    """
    extn=filename.rsplit('.',1)[-1]
    if extn=='t7':
        return cv2.dnn.readNetFromTorch(filename)
    if extn=='xml':
        return cv2.dnn.readNetFromONNX(filename)
    raise IOError('Unknown file format for "'+filename+'"')


def styleTransfer(image,styleNetwork):
    """
    Use a neural network to apply a style to an image

    :param styleNetwork: can be a loaded cv Net object, or a filename

    Sample style transfer models:
        http://cs.stanford.edu/people/jcjohns/fast-neural-style/models/
            /instance_norm/candy.t7
            /instance_norm/la_muse.t7
            /instance_norm/mosaic.t7
            /instance_norm/feathers.t7
            /instance_norm/the_scream.t7
            /instance_norm/udnie.t7"
            /eccv16/the_wave.t7
            /eccv16/starry_night.t7
            /eccv16/la_muse.t7
            /eccv16/composition_vii.t7
    """
    if isinstance(styleNetwork,str):
        styleNetwork=loadNeuralNetwork(styleNetwork)
    # load the input image, resize it to have a width of 600 pixels, and
    # then grab the image dimensions
    image = numpyArray(image)
    #image = imutils.resize(image, width=600)
    image=crop(image,(599,599))
    imgSize = image.shape[:2]
    print(image.shape,image[0,0,0])
    # construct a blob from the image, set the input, and then perform a
    # forward pass of the network
    blob = cv2.dnn.blobFromImage(image, 1.0,imgSize,
        (103.939, 116.779, 123.680), swapRB=False, crop=False)
    styleNetwork.setInput(blob)
    #start = time.time()
    output = styleNetwork.forward()
    #end = time.time()
    # reshape the output tensor, add back in the mean subtraction, and
    # then swap the channel ordering
    output = output.reshape((3, output.shape[2], output.shape[3]))
    output[0] += 103.939
    output[1] += 116.779
    output[2] += 123.680
    output /= 255.0
    output = output.transpose(1, 2, 0)
    # This is for testing
    cv2.imshow("Input", image)
    cv2.imshow("Output", output)
    cv2.waitKey(0)
    return output # TODO: reformat first


def cmdline(args):
    """
    Run the command line

    :param args: command line arguments (WITHOUT the filename)
    """
    import os
    printhelp=False
    if not args:
        printhelp=True
    else:
        img=None
        for arg in args:
            if arg.startswith('-'):
                arg=[a.strip() for a in arg.split('=',1)]
                if arg[0] in ['-h','--help']:
                    printhelp=True
                elif arg[0]=='-styleTransfer':
                    if img is None:
                        raise Exception('ERR: no image to modify')
                    if len(arg)<2:
                        raise Exception('ERR: no style specified')
                    img=styleTransfer(img,os.path.abspath(arg[1]))
                else:
                    print('ERR: unknown argument "'+arg[0]+'"')
            else:
                img=arg
    if printhelp:
        print('Usage:')
        print('  neural.py file [options]')
        print('Options:')
        print('   -styleTransfer=net ...... do a style transfer based on a saved neural network')
        print('Notes:')
        print('   supports .t7 torch neural networks and ONNX xml networks')


if __name__=='__main__':
    import sys
    cmdline(sys.argv[1:])

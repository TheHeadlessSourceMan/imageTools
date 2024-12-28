#!/usr/bin/env
# -*- coding: utf-8 -*-
"""
This automatically detects regions of interest in an image.
"""
try:
    # first try to use bohrium, since it could help us accelerate
    # https://bohrium.readthedocs.io/users/python/
    import bohrium as np # type: ignore
except ImportError:
    # if not, plain old numpy is good enough
    import numpy as np
try:
    import opencv # type: ignore # pylint: disable=unused-import
    has_opencv=True
except ImportError:
    has_opencv=False
import scipy

from featureFinder import (
    selectSkin,selectEyes,selectFaces,selectBackground,selectHair)
from helper_routines import normalize
from colorSpaces import getChannel,grayscale
import imageRepr
#from .featureFinder import *
#from .colorSpaces import *
#from .helper_routines import *
#from . import resizing


def selectHighContrast(image):
    """
    get mask based on very bright or very dark areas

    returns b&w image where white=interesting, black=uninteresting

    NOTE: since we are using a color-weighted grayscale conversion,
        this does account somewhat for color discrimination of human vision.
    """
    image=grayscale(image,method='BT.709')
    a=np.average(image)
    return normalize(np.abs(image-a))


def selectHighSaturation(image,power=2):
    """
    get mask based on very bright or very dark areas

    returns b&w image where white=interesting, black=uninteresting
    """
    return normalize(np.power(getChannel(image,'S'),power))


def selectSobel(image):
    """
    get mask based seam edges

    returns b&w image where white=interesting, black=uninteresting
    """
    img=normalize(grayscale(image))
    x=scipy.ndimage.sobel(img,1)
    y=scipy.ndimage.sobel(img,0)
    mag=np.hypot(x,y)
    return normalize(mag)


def interest(image):
    """
    get regions of interest in the image

    returns b&w image where white=interesting, black=uninteresting
    """
    program=[] # (weight, function)
    if not has_opencv:
        print('WARN: Attempting auto-detection of regions of interest.')
        print('\t  This works A LOT better with OpenCV installed.')
        print('\t  try: pip install opencv-contrib-python')
        print('\t  or go here:')
        print('\t    https://sourceforge.net/projects/opencvlibrary/files/')
        program=[
            (0.5,selectSkin),
            (0.1,selectHighContrast),
            (0.1,selectHighSaturation),
            (0.05,selectSobel)]
    else:
        program=[
            (0.8,selectEyes),
            (0.8,selectFaces),
            (0.5,selectSkin),
            (0.3,selectHighContrast),
            (0.5,selectHighSaturation),
            (0.3,selectSobel)
            ]
    image=imageRepr.numpyArray(image)
    outImage=np.zeros(image.shape[0:2])
    totalWeight=0
    for weight,_ in program:
        totalWeight+=weight
    for weight,fn in program:
        weight=weight/totalWeight
        if weight<0.001:
            #print("skipping",fn.__name__,weight)
            continue
        #print("calculating",fn.__name__,weight)
        im=fn(image)
        if weight!=1.0:
            im=im*weight
        outImage=outImage+im
    return normalize(outImage)


def cmdline(args):
    """
    Run the command line

    :param args: command line arguments (WITHOUT the filename)
    """
    from helper_routines import preview
    printhelp=False
    if not args:
        printhelp=True
    else:
        image=None
        for arg in args:
            if arg.startswith('-'):
                arg=[a.strip() for a in arg.split('=',1)]
                if arg[0] in ['-h','--help']:
                    printhelp=True
                elif arg[0]=='--interest':
                    image=interest(image)
                elif arg[0]=='--selectFaces':
                    image=normalize(selectFaces(image))
                elif arg[0]=='--selectEyes':
                    image=normalize(selectEyes(image))
                elif arg[0]=='--selectBackground':
                    image=normalize(selectBackground(image))
                elif arg[0]=='--selectHair':
                    image=normalize(selectHair(image))
                elif arg[0]=='--selectSkin':
                    image=normalize(selectSkin(image))
                elif arg[0]=='--selectHighContrast':
                    image=normalize(selectHighContrast(image))
                elif arg[0]=='--selectHighSaturation':
                    image=normalize(selectHighSaturation(image))
                elif arg[0]=='--selectSobel':
                    image=normalize(selectSobel(image))
                elif arg[0]=='--show':
                    preview(image)
                elif arg[0]=='--save':
                    if len(arg)>1:
                        imageRepr.pilImg(image).save(arg[1])
                else:
                    print('ERR: unknown argument "'+arg[0]+'"')
            else:
                image=arg
    if printhelp:
        print('Usage:')
        print('  autoInterest.py image.jpg [options]')
        print('Options:')
        print('   --selectFaces ........... select human faces in an image')
        print('   --selectEyes ............ select eyes in an image')
        print('   --selectBackground ...... select the background of an image')
        print('   --selectHair ............ select hair in an image')
        print('   --selectSkin ............ select human skin in an image')
        print('   --selectHighContrast .... select very bright or very dark')
        print('                             areas of an image')
        print('   --selectHighSaturation .. select vibrant colors in an image')
        print('   --selectSobel ........... select edge strength in an image')
        print('   --interest .............. region of interest for an image')
        print('   --show .................. show the working image')
        print('   --save=filename ......... save the working image')


if __name__=='__main__':
    import sys
    cmdline(sys.argv[1:])

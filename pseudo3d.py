#!/usr/bin/env
# -*- coding: utf-8 -*-
"""
Tools to create/apply pseudo 3d data, including relighting.
"""
import math
try:
    # first try to use bohrium, since it could help us accelerate
    # https://bohrium.readthedocs.io/users/python/
    import bohrium as np # type: ignore
except ImportError:
    # if not, plain old numpy is good enough
    import numpy as np
import scipy.ndimage # type: ignore
from .helper_routines import normalize
from .imageRepr import numpyArray
from .colors import strToColor
from .resizing import crop
from .colorSpaces import grayscale


def normalMapFromImage(img,zPower:float=0.33):
    """
    Attempt to get a normal map from an image using edge detection.

    :param img: the image to create a normal map for
    :param zPower: percent of z estimation to allow (keep it low because this
        is always a total guess)

    NOTE: This is a common approach, but it is only an approximation, not
        actual height data. It will only give you something that may or may
        not look 3d. The real way to do this is to directly measure the height
        data directly using something like a kinect or 3d digitizer. An
        acceptable alternative is a multi-angle photo stitcher such as
        123D Catch. In fact, if all you have is a 2d image, something like
        AwesomeBump would be more versitile.
            https://github.com/kmkolasinski/AwesomeBump

    See also:
        https://github.com/RobertBeckebans/gimp-plugin-insanebump
    """
    img=normalize(grayscale(img))
    x=scipy.ndimage.sobel(img,1)
    y=scipy.ndimage.sobel(img,0)
    mag=np.hypot(x,y)
    z=normalize(scipy.ndimage.distance_transform_edt(mag))*\
        zPower+(0.5-zPower/2)
    return np.dstack((x,y,z))


def directionToNormalColor(
    azimuth:float,
    elevation:float
    )->np.ndarray:
    """
    given a direction, convert it into an RGB normal color

    :param azimuth: compass direction
    :param elevation: up/down direction
    """
    azimuth=math.radians(float(azimuth))
    elevation=math.radians(float(elevation))
    return (
        np.array([
            math.sin(azimuth),
            math.cos(azimuth),
            math.cos(elevation)
            ])+1
        )/2


def applyDirectionalLight(
    img,
    normalMap=None,
    lightAzimuth:float=45,
    lightElevation:float=45,
    lightColor=None):
    """
    Add a directional light to the image.

    :param img: the image to light
    :param normalMap: a normal map to define its shape
        (if None, use normalMapFromImage(img))
    :param lightAzimuth: compass direction of the light source
    :param lightElevation: up/down direction of the light source
    :param lightColor: color of the light source (if None, use white)

    TODO: doesn't work the greatest
    """
    img=numpyArray(img)
    if lightColor is None:
        lightColor=[1.0,1.0,1.0]
    if normalMap is None:
        normalMap=normalMapFromImage(img)
    else:
        normalMap=numpyArray(normalMap)
        normalMap=crop(normalMap,img)
    lightColor=strToColor(lightColor)
    dLight=directionToNormalColor(lightAzimuth,lightElevation)
    normalMap=normalMap[:,:]*dLight
    lightMap=np.average(normalMap,axis=-1)
    lightMap=lightMap[:,:,np.newaxis]*lightColor
    return img+lightMap


def heightMapFromNormalMap(normalMap):
    """
    Create a height map (sometimes called a bumpMap) from a normalMap

    :param normalMap: normal map to convert

    NOTE: Essentially, this is just the blue channel pointing towards us.
    """
    normalMap=numpyArray(normalMap)
    return normalize(normalMap[:,:,2])


def normalMapFromHeightMap(heightMap):
    """
    Create a normal map from a height map (sometimes called a bumpMap)

    :param heightMap: height map to convert

    Comes from:
        http://www.juhanalankinen.com/calculating-normalmaps-with-python-and-numpy/
    """
    heightMap=numpyArray(heightMap)
    heightMap=normalize(heightMap)
    matI = np.identity(heightMap.shape[0]+1)
    matNeg = -1 * matI[0:-1, 1:]
    matNeg[0, -1] = -1
    matI = matI[1:, 0:-1]
    matI[-1,0] = 1
    matI += matNeg
    matGradX = (matI.dot(heightMap.T)).T
    matGradY = matI.dot(heightMap)
    matNormal = np.zeros([heightMap.shape[0], heightMap.shape[1], 3])
    matNormal[:,:,0] = -matGradX
    matNormal[:,:,1] = matGradY
    matNormal[:,:,2] = 1.0
    fNormMax = np.max(np.abs(matNormal))
    matNormal = ((matNormal / fNormMax) + 1.0) / 2.0
    return matNormal


def cmdline(args):
    """
    Run the command line

    :param args: command line arguments (WITHOUT the filename)
    """
    printHelp=False
    if not args:
        printHelp=True
    else:
        from imageRepr import pilImage
        from helper_routines import preview,clampImage
        img=None
        norm=None
        bump=None
        for arg in args:
            if arg.startswith('-'):
                arg=[a.strip() for a in arg.split('=',1)]
                if arg[0] in ['-h','--help']:
                    printHelp=True
                elif arg[0]=='--img':
                    img=arg[1]
                elif arg[0]=='--norm':
                    bump=None
                    norm=arg[1]
                elif arg[0]=='--bump':
                    norm=None
                    bump=arg[1]
                elif arg[0]=='--addDirectional':
                    if len(arg)<2:
                        params=[]
                    else:
                        params=arg[1].split(',',3)
                        if norm is None:
                            if bump is not None:
                                norm=normalMapFromHeightMap(bump)
                    img=applyDirectionalLight(img,norm,*params)
                elif arg[0]=='--show':
                    preview(img)
                elif arg[0]=='--showNorm':
                    if norm is None:
                        if bump is None:
                            norm=normalMapFromImage(img)
                        else:
                            norm=normalMapFromHeightMap(bump)
                    preview(norm)
                elif arg[0]=='--showBump':
                    if bump is None:
                        if norm is None:
                            norm=normalMapFromImage(img)
                        bump=heightMapFromNormalMap(norm)
                    preview(bump)
                elif arg[0]=='--save':
                    pilImage(clampImage(img)).save(arg[1])
                elif arg[0]=='--saveNorm':
                    if norm is None:
                        if bump is None:
                            norm=normalMapFromImage(img)
                        else:
                            norm=normalMapFromHeightMap(bump)
                    pilImage(clampImage(norm)).save(arg[1])
                elif arg[0]=='--saveBump':
                    if bump is None:
                        if norm is None:
                            norm=normalMapFromImage(img)
                        bump=heightMapFromNormalMap(norm)
                    pilImage(clampImage(bump)).save(arg[1])
                else:
                    print('ERR: unknown argument "'+arg[0]+'"')
            else:
                print('ERR: unknown argument "'+arg+'"')
    if printHelp:
        print('Usage:')
        print('  pseudo3d.py [options]')
        print('Options:')
        print('   --img=[filename] ....... set the image')
        print('   --norm=[filename] ...... set the normal map')
        print('   --bump=[filename] ...... set the bump map')
        print('   --addDirectional=azimuth,elevation,color')
        print('   --show ................. show the image')
        print('   --showNorm ............. show the normal map')
        print('   --showBump ............. show the bump map')
        print('   --save=[filename] ...... save out the result')
        print('   --saveNorm=[filename] .. save out the normal map')
        print('   --saveBump=[filename] .. save out the bump map')


if __name__=='__main__':
    import sys
    cmdline(sys.argv[1:])

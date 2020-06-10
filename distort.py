#!/usr/bin/env
# -*- coding: utf-8 -*-
"""
This contains image warping/distorting tools

See also:
    http://www.imagemagick.org/Usage/distorts
"""
import math
from collections import namedtuple
try:
    # first try to use bohrium, since it could help us accelerate
    # https://bohrium.readthedocs.io/users/python/
    import bohrium as np
except ImportError:
    # if not, plain old numpy is good enough
    import numpy as np
import scipy
from .imageRepr import *
from .colorSpaces import *
from .resizing import *
from .helper_routines import clampImage


Pin=namedtuple('Pin','fromLocation toLocation fromRadius toRadius strength')
def morph(img,pins):
    """
    Perform an image "morph" where a series of pins are moved
    to stretch the image around. This can also achieve animation similar to the
    Adobe AfterEffects "puppet tool".

    This works by computing a distance for each pin to get it's power
    mapping.  Then multiply that by the transform.

    TODO: or should I simply use scipy.ndimage.map_coordinates
        https://stackoverflow.com/questions/25169126/image-warping-with-python
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html#scipy-ndimage-map-coordinates

    :param img: can be a pil image, numpy array, etc
    :param pins: [(fromLocation,toLocation,fromRadius=1,toRadius=fromRadius,strength=1.0)]
    """
    for pin in pins:
        raise NotADirectoryError()
    return img

def pinchPull(img,amount,location=None,radius=None):
    """
    Perform an image "pinch/pull" to grow or shrink a region of the image.

    Common uses:
        1) Correct camera's "barrel distortion"
        2) A "goo" tool for making funny photos

    NOTE: this is more or less a single-point morph, where the point doesn't
        even move!

    :param img: can be a pil image, numpy array, etc
    :param amount: in terms how many pixels one pixel at the center becomes
        (that is, <1.0 is a pinch)
    :param location: where to perform the operation (default=center)
    :param radius: how large the transform (default=center to any corner)
    """
    if location is None:
        location=(image.width/2,image.height/2)
    if radius is None:
        radius=math.sqrt((image.width/2)^2+(image.height/2)^2)
    pins=[(location,location,radius,radius*amount)]
    return morph(img,pins)

def bruteDisplaceImage(img,displacementMap,distance=10,angle=45):
    """
    this version is much faster even with stupid loops, but the problem
    is it leads to pixelization because it's simply moving from point a to point b
    """
    angle=math.radians(angle)
    sa=math.sin(angle)
    ca=math.cos(angle)
    img=numpyArray(img)
    displacementMap=numpyArray(displacementMap)
    maxDisp=distance
    if len(displacementMap.shape)>2:
        # convert displacement map to grayscale
        displacementMap=grayscale(displacementMap)
    #bg=np.zeros((img.shape[0]+maxDisp*2,img.shape[1]+maxDisp*2,4))
    bg=np.copy(img)
    img=numpyArray(changeMode(img,imageMode(bg)))
    for x in range(min(img.shape[0],displacementMap.shape[0])):
        for y in range(min(img.shape[1],displacementMap.shape[1])):
            delta=displacementMap[x,y]*distance
            bg[int(round(x+delta*sa-maxDisp)),int(round(y-delta*ca-maxDisp))]=img[x,y]
    return bg

def displaceImage(img,displacementMap,distance=10,angle=45):
    """

    NOTE: If displacementMap is not grayscale, convert it using the default (BT.709)

    NOTE: If displacementMap is smaller than img, then everthing out of bounds
    is not displaced!
    """
    angle=math.radians(angle)
    img=numpyArray(img)
    displacementMap=numpyArray(displacementMap)
    maxDisp=0#distance
    sa=math.sin(angle)
    ca=math.cos(angle)
    if len(displacementMap.shape)>2:
        displacementMap=grayscale(displacementMap)
    def dissp(point):
        """
        Called for each point in the array.  Returns a same shaped tuple where it should go.
        """
        if point[0]<displacementMap.shape[0] and point[1]<displacementMap.shape[1]:
            delta=displacementMap[point[0],point[1]]*distance
            if len(point)>2: # array is (w,h,rgb)
                point=(point[0]+delta*sa-maxDisp,point[1]-delta*ca-maxDisp,point[2])
            else: # array is (w,h) -- aka grayscale
                point=(point[0]+delta*sa-maxDisp,point[1]-delta*ca-maxDisp)
        return point
    #bg=np.zeros((img.shape[0],img.shape[1],3))
    bg=np.copy(img)
    # TODO: for some reason this only works with order<3
    img=scipy.ndimage.geometric_transform(img,dissp,order=2,output=bg,mode="nearest")
    return clampImage(img)


def warpCircle(img):
    """
    wrap the image around a circle

    NOTE: only row 0 is used
    """
    size=getSize(img)
    center=(size[0]/2,size[1]/2)
    def circ(point):
        r=math.sqrt((point[0]-center[0])**2+(point[1]-center[1])**2)
        return (0,r)
    return scipy.ndimage.geometric_transform(img,circ,mode='nearest')


def warpRotate(img,theta):
    """
    wrap the image around a circle

    NOTE: only row 0 is used

    TODO: need to resize img larger to encoumpass rotated bounds
    """
    originalSize=getSize(img)
    rot=scipy.ndimage.interpolation.rotate(img,theta)
    newSize=getSize(rot)
    bounds=[
        (newSize[0]-originalSize[0])/2,
        (newSize[1]-originalSize[1])/2,
        originalSize[0],originalSize[1]]
    return crop(rot,bounds)


def cmdline(args):
    """
    Run the command line

    :param args: command line arguments (WITHOUT the filename)
    """
    printhelp=False
    if not args:
        printhelp=True
    else:
        for arg in args:
            if arg.startswith('-'):
                arg=[a.strip() for a in arg.split('=',1)]
                if arg[0] in ['-h','--help']:
                    printhelp=True
                else:
                    print(('ERR: unknown argument "'+arg[0]+'"'))
            else:
                print(('ERR: unknown argument "'+arg+'"'))
    if printhelp:
        print('Usage:')
        print('  distort.py [options]')
        print('Options:')
        print('   NONE')


if __name__=='__main__':
    import sys
    cmdline(sys.argv[1:])

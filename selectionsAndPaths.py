#!/usr/bin/env
# -*- coding: utf-8 -*-
"""
A colleciton of routines to work with selection in bitmaps and vector paths
"""

try:
    # first try to use bohrium, since it could help us accelerate
    # https://bohrium.readthedocs.io/users/python/
    import bohrium as np
except ImportError:
    # if not, plain old numpy is good enough
    import numpy as np
import scipy.ndimage
from PIL import Image, ImageDraw
from .colors import *
from .helper_routines import *


def selectByColor(img,color,tolerance=0,soften=10,smartSoften=True,colorspace='RGB'):
    """
    Select all pixels of a given color

    :param img: can be a pil image, numpy array, etc
    :param color: (in colorspace units) can be anything that pickColor returns:
        a color int array
        a color string to decode
        a color range (colorLow,colorHigh)
        an array of any of these
    :param tolerance: how close the selection must be
    :param soften: apply a soften radius to the edges of the selection
    :param smartSoften: multiply the soften radius by how near the pixel is to the selection color
    :param colorspace: change the given img (assumed to be RGB) into another corlorspace before matching

    :returns: black and white image in a numpy array (usable as a selection or a mask)
    """
    from . import colorSpaces
    img=numpyArray(img)
    img=colorSpaces.changeColorspace(img,colorspace)
    if (isinstance(color,list) and isinstance(color[0],list)) or (isinstance(color,np.ndarray) and len(color.shape)>1):
        # there are multiple colors, so select them all one at a time
        # TODO: this could possibly be made faster with array operations??
        ret=None
        for c in color:
            if ret is None:
                ret=selectByColor(img,c,tolerance,soften,smartSoften)
            else:
                ret=ret+selectByColor(img,c,tolerance,soften,smartSoften)
        ret=clampImage(ret)
    elif isinstance(color,tuple):
        # color range - select all colors "between" these two in the given color space
        color=(matchColorToImage(color[0],img),matchColorToImage(color[1],img))
        matches=np.logical_and(img>=color[0],img<=color[1])
        ret=np.where(matches.all(axis=2),1.0,0.0)
    else:
        # a single color (or a series of colors that have been averaged down to one)
        color=matchColorToImage(color,img)
        if isFloat(color):
            imax=1.0
            imin=0.0
        else:
            imax=255
            imin=0
        numColors=img.shape[-1]
        avgDelta=np.sum(np.abs(img[:,:]-color),axis=-1)/numColors
        ret=np.where(avgDelta<=tolerance,imax,imin)
        if soften>0:
            ret=gaussianBlur(ret,soften)
            if smartSoften:
                ret=np.minimum(imax,ret/avgDelta)
    return ret


def selectByPoint(img,location,tolerance=0,soften=10,smartSoften=True,colorspace='RGB',pickMode='average'):
    """
    Works like the "magic wand" selection tool.
    It is different than selectByColor(img,pickColor(img,location)) in that only a contiguious
        region is selected

    :param img: can be a pil image, numpy array, etc
    :param location: can be a single point or an [x,y,w,h]
    :param tolerance: how close the selection must be
    :param soften: apply a soften radius to the edges of the selection
    :param smartSoften: multiply the soften radius by how near the pixel is to the selection color
    :param colorspace: change the given img (assumed to be RGB) into another corlorspace before matching
        TODO: implement this!

    :returns: black and white image in a numpy array (usable as a selection or a mask)
    """
    img=numpyArray(img)
    img=changeColorspace(img)
    # selectByColor, but the
    selection=selectByColor(img,pickColor(img,location,pickMode),tolerance,soften,smartSoften,colorspace=imageMode(img))
    # now identify islands from selection
    labels,_=scipy.ndimage.label(selection)
    # grab only the named islands within the original location
    if len(location)<4:
        labelsInSel=[labels[location[0],location[1]]]
    else:
        labelsInSel=np.unique(labels[location[0]:location[0]+location[2]+1,location[0]:location[0]+location[2]+1])
    # only keep portions of selection within our islands
    selection=np.where(np.isin(labels,labelsInSel),selection,0.0)
    return selection


def gaussianBlur(img,sizeX,sizeY=None,edge='clamp'):
    """
    perform a gaussian blur on an image

    :param img: can be a pil image, numpy array, etc
    :param sizeX: the x radius of the blur
    :param sizeY: the y radius of the blur (if None, use same as X)
    :param edge: what to do for more pixels when we reach an edge
        can be: "clamp","mirror","wrap", or a color
        default is "clamp"

    :returns: image in a numpy array
    """
    if sizeY is None:
        sizeY=sizeX
    if edge=='clamp':
        mode='nearest'
        cval=0.0
    elif edge=='mirror':
        mode='mirror'
        cval=0.0
    elif edge=='wrap':
        mode='wrap'
        cval=0.0
    else:
        mode='constant'
        cval=np.array(strToColor(edge))
    img=numpyArray(img)
    img=scipy.ndimage.gaussian_filter(img,(sizeY,sizeX),mode=mode,cval=cval)
    return img


def morphology(func,img,amount,edge='clamp',shape=None):
    """
    perform a morphological operation on an image
        "open","close","dilate",or "erode"

    For a definition of what those things mean:
        https://en.wikipedia.org/wiki/Mathematical_morphology

    :param func: morphological function to perform
    :param img: can be a pil image, numpy array, etc
    :param amount: how wide in pixels the operator is
    :param edge: what to do for more pixels when we reach an edge
        can be: "clamp","mirror","wrap", or a color
        default is "clamp"
    :param shape: a b+w shape to sweep along the img outline
        disables amount value

    :returns: image in a numpy array
    """
    if edge=='clamp':
        mode='nearest'
        cval=0.0
    elif edge=='mirror':
        mode='mirror'
        cval=0.0
    elif edge=='wrap':
        mode='wrap'
        cval=0.0
    else:
        mode='constant'
        cval=np.array(strToColor(edge))
    amount=int(amount) # they don't like floats
    if amount<0:
        # if they specify a negative amount, flip the operation!
        amount=-amount
        if func=='dilate':
            func='erode'
        elif func=='erode':
            func='dilate'
        elif func=='open':
            func='close'
        elif func=='close':
            func='open'
    img=numpyArray(img)
    if func=='erode':
        img=scipy.ndimage.grey_erosion(img,(amount),shape,mode=mode,cval=cval)
    elif func=='dilate':
        img=scipy.ndimage.grey_dilation(img,(amount),shape,mode=mode,cval=cval)
    elif func=='open':
        img=scipy.ndimage.grey_opening(img,(amount),shape,mode=mode,cval=cval)
    elif func=='close':
        img=scipy.ndimage.grey_closing(img,(amount),shape,mode=mode,cval=cval)
    elif func in ('gradient','outline'):
        img=scipy.ndimage.morphological_gradient(img,(amount),shape,mode=mode,cval=cval)
    elif func in ('tophat','whitehat'):
        img=scipy.ndimage.white_tophat(img,(amount),shape,mode=mode,cval=cval)
    elif func=='blackhat':
        img=scipy.ndimage.black_tophat(img,(amount),shape,mode=mode,cval=cval)
    return img


def selectionToPath(img,midpoint=0.5):
    """
    convert a black and white selection (or mask) into a path

    :param img: a pil image, numpy array, etc
    :param midpoint: for grayscale images, this is the cuttoff point for yes/no

    :return: a closed polygon [(x,y)]

    TODO: I resize the image to give laplace a border to lock onto, but
        do I need to scale the points back down again??
    """
    img=imageBorder(img,1,0)
    img=numpyArray(img)
    img=np.where(img>=midpoint,1.0,0.0)
    img=scipy.ndimage.laplace(img)
    points=np.nonzero(img) # returns [[y points],[x points]]
    points=np.transpose(points)
    # convert to a nice point-pair representation
    points=[(point[1],point[0]) for point in points]
    return points


def pathToSelection(path,filled=True,outlineWidth=0):
    """
    by using fill=False and outlineWidth=n you can create a border selection
    """
    if outlineWidth>0:
        borderColor=1
    else:
        borderColor=0
    if filled:
        backgroundColor=1
    else:
        backgroundColor=0
    return renderPath(path,None,1,borderColor,backgroundColor)


def renderPath(path,intoImg=None,borderSize=1,borderColor=None,backgroundColor=None):
    """
    render a path into an image

    :param intoImg: if not specified, a new one will be created
    :param borderSize: how thick the border is (TODO: still need to implement)
    :param borderColor: color of the border
    :param backgroundColor: color of the background
    """
    if intoImg is None:
        w=0
        h=0
        for pt in path:
            if pt[0]>w:
                w=pt[0]
            if pt[1]>h:
                h=pt[1]
        intoImg=Image.new('L',(w,h),0)
        if borderColor is None:
            borderColor=1
        if backgroundColor is None:
            backgroundColor=0
    else:
        if borderColor is None:
            borderColor=(255,0,0,128)
        if backgroundColor is None:
            backgroundColor=(255,0,0,128)
    ImageDraw.Draw(intoImg).polygon(path,outline=borderColor,fill=backgroundColor)
    return intoImg


def cmdline(args):
    """
    Run the command line

    :param args: command line arguments (WITHOUT the filename)
    """
    printhelp=False
    if not args:
        printhelp=True
    else:
        selection=None
        selectionTolerance=0.10
        soften=0
        smartSoften=True
        pickMode='average'
        img=None
        for arg in args:
            if arg.startswith('-'):
                arg=[a.strip() for a in arg.split('=',1)]
                if arg[0] in ['-h','--help']:
                    printhelp=True
                elif arg[0]=='--selectionTolerance':
                    selectionTolerance=float(arg[1])
                elif arg[0]=='--selectByColor':
                    selection=selectByColor(img,arg[1],selectionTolerance,soften,smartSoften)
                elif arg[0]=='--selectByPoint':
                    selection=selectByPoint(img,[int(x) for x in arg[1].split(',')],selectionTolerance,soften,smartSoften,pickMode=pickMode)
                elif arg[0]=='--pickMode':
                    pickMode=arg[1]
                elif arg[0]=='--soften':
                    soften=float(arg[1])
                elif arg[0]=='--smartSoften':
                    smartSoften=arg[1][0] in ['t','T','y','Y','1']
                elif arg[0]=='--pickColor':
                    color=pickColor(img,[int(x) for x in arg[1].split(',')],pickMode)
                    print((colorToStr(color)))
                elif arg[0]=='--showSelection':
                    pilImage(selection).show()
                elif arg[0]=='--showPath':
                    points=selectionToPath(selection)
                    iout=img.copy()
                    iout=renderPath(points,iout)
                    pilImage(iout).show()
                elif arg[0] in ['--erode','--dilate','--open','--close']:
                    selection=morphology(arg[0][2:],selection,float(arg[1]))
                else:
                    print(('ERR: unknown argument "'+arg[0]+'"'))
            else:
                img=defaultLoader(arg)
    if printhelp:
        print('Usage:')
        print('  selectionsAndPaths.py image.jpg [options]')
        print('Options:')
        print('   --selectionTolerance=0.5 . set tolerance for subsequent selection operations')
        print('   --selectByColor=RGB ...... select by a given color')
        print('   --soften=r ............... radius to soften the selection')
        print('   --smartSoften=true ....... modify softenss based on similarity')
        print('   --pickMode=mode .......... when picking an area "average","all",or "range"')
        print('   --pickColor=x,y[,w,h] .... pick colors')
        print('   --showSelection .......... show only the selected portion of the image')
        print('   --showPath ............... show selection as a path on top of the original image')
        print('   --erode=amt .............. erode the selection')
        print('   --dilate=amt ............. dilate the selection')
        print('   --open=amt ............... "open" the selection')
        print('   --close=amt .............. "close" the selection')
        print('Notes:')
        print('   * All filenames can also take file:// http:// https:// ftp:// urls')


if __name__=='__main__':
    import sys
    cmdline(sys.argv[1:])

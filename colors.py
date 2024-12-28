#!/usr/bin/env
# -*- coding: utf-8 -*-
"""
Pick, manage, and transform colors to/from html.
"""
try:
    # first try to use bohrium, since it could help us accelerate
    # https://bohrium.readthedocs.io/users/python/
    import bohrium as np # type: ignore
except ImportError:
    # if not, plain old numpy is good enough
    import numpy as np
#from .colorTools import
from .imageRepr import numpyArray,isFloat


def strToColor(s,asFloat=True,defaultColor=None):
    """
    convert an html color spec to a color array

    always returns an rgba[], regardless of input color being:
        #FFFFFF
        #FFFFFFFF
        rgb(128,12,23)
        rgba(234,33,23,0)
        yellow

    :param s: an html color string
    :param asFloat: always return a 0.0 to 1.0 floats verses a 0 to 255 bytes

    :returns: an rgba[]
    """
    c=Color()
    c.fromHtml(s,asFloat,defaultColor)
    return c


def colorToStr(s,preferHex=True,alwaysHex=False,useNamed=False):
    """
    :param s: the color to convert (If it is a string, it will be converted
        to a color, then back.  Thus you can convert from one string
        format to another.)
    :param preferHex: return hex over rgb() but NOT over rgba()
    :param alwaysHex: always returns a hex string over rgb() or even rgba()
    :param useNamed: before anything else, check to see if there is
        a named color that matches
    """
    if not isinstance(s,Color):
        s=Color(s)
    return s.toHtml(preferHex,alwaysHex,useNamed)


def pickColor(img,location,pickMode='average'):
    """
    pick a color from an image at a given location

    :param img: can be a pil image, numpy array, etc
    :param location: can be a single point or an [x,y,w,h] to average
    :param pickMode: what to do with multiple pixels
        choices are average,range,or all
    :return:
        if a single point or "average", return one color
        if "range" return (minColor,maxColor)
        if "all" return [all colors]
    """
    ret=None
    img=numpyArray(img)
    if len(location)<4:
        # if it's a single pixel, easy peasy
        ret=img[location[0],location[1]]
    else:
        region=img[
            location[0]:location[0]+location[2]+1,
            location[0]:location[0]+location[2]+1]
        if pickMode=='average':
            ret=[]
            for i in range(img.shape[-1]):
                ret.append(Color(np.mean(region[:,:,i])))
        elif pickMode=='range':
            ret=(
                Color(np.min(region,axis=(0,1))),
                Color(np.max(region,axis=(0,1))))
        elif pickMode=='all':
            ret=region.reshape(-1,region.shape[-1])
            ret=np.unique(ret,axis=0)
            ret=[Color(r) for r in ret]
    return ret


def matchColorToImage(color,img):
    """
    match the color to the image mode
    """
    imgChan=img.shape[-1]
    if isinstance(color,(float,int)):
        if isFloat(img):
            if isinstance(color,int):
                color=color/255.0
        elif isinstance(color,int):
            color=int(color*255)
    else:
        colChan=len(color)
        if imgChan>colChan:
            msg="Don't know how to convert color[%d] to image pixel[%d]"%(len(color),img.shape[-1]) # noqa: E501 # pylint: disable=line-too-long
            raise NotImplementedError(msg)
        if imgChan<colChan:
            color=color[0:imgChan]
        imgFloat=isFloat(img)
        colFloat=isFloat(color)
        if imgFloat!=colFloat:
            if imgFloat:
                color=np.array(color)/255.0
            else:
                color=np.int(np.array(color)*255)
    return color


def cmdline(args):
    """
    Run the command line

    :param args: command line arguments (WITHOUT the filename)
    """
    printHelp=False
    if not args:
        printHelp=True
    else:
        for arg in args:
            if arg.startswith('-'):
                arg=[a.strip() for a in arg.split('=',1)]
                if arg[0] in ['-h','--help']:
                    printHelp=True
                else:
                    print('ERR: unknown argument "'+arg[0]+'"')
            else:
                print('ERR: unknown argument "'+arg+'"')
    if printHelp:
        print('Usage:')
        print('  colors.py [options]')
        print('Options:')
        print('   NONE')


if __name__=='__main__':
    import sys
    cmdline(sys.argv[1:])

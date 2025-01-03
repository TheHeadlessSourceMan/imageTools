#!/usr/bin/env
# -*- coding: utf-8 -*-
"""
Routines to convert between color spaces
"""
try:
    # first try to use bohrium, since it could help us accelerate
    # https://bohrium.readthedocs.io/users/python/
    import bohrium as np # type: ignore
except ImportError:
    # if not, plain old numpy is good enough
    import numpy as np
from .imageRepr import numpyArray,imageMode,pilImage


def changeColorspace(img,toSpace='RGB',fromSpace=None):
    """
    change from any colorspace to any colorspace

    :param img: the image to convert
    :param toSpace: space to convert to
    :param fromSpace: space to convert from - guess from img if None

    NOTE: for more colorspace conversions than you can shake a chameleon at:
        http://colormine.org/
    """
    img=numpyArray(img)
    if fromSpace is None:
        fromSpace=imageMode(img)
    raiseErr=False
    if fromSpace!=toSpace:
        if fromSpace=='RGB':
            if toSpace=='HSV':
                img=rgb2hsvArray(img)
            elif toSpace=='CMYK':
                img=rgb2cmykArray(img)
            else:
                raiseErr=True
        elif fromSpace=='CMYK':
            if toSpace=='HSV':
                img=rgb2hsvArray(cmyk2rgbArray(img))
            elif toSpace=='RGB':
                img=cmyk2rgbArray(img)
            else:
                raiseErr=True
        elif fromSpace=='HSV':
            if toSpace=='RGB':
                img=hsv2rgbArray(img)
            elif toSpace=='CMYK':
                img=rgb2cmykArray(hsv2rgbArray(img))
            else:
                raiseErr=True
        else:
            raiseErr=True
    if raiseErr:
        msg='No conversion exists from colorspace "'+fromSpace+'" to "'+toSpace+'"' # noqa: E501 # pylint: disable=line-too-long
        raise NotImplementedError(msg)
    return img


def rgb2hsvImage(img):
    """
    same as rgb2hsvArray only returns an image

    :param img: image to convert.  can be pil image or numpy array

    :return img: returns an "rgb" image, that is actually "hsv"
    """
    return pilImage(rgb2hsvArray(img))


def rgb2hsvArray(rgb):
    """
    Transform an rgb array to an hsv array

    :param rgb: the input image.  can be pil image or numpy array
    :return hsv: the output array

    This comes from scikit-image:
        https://github.com/scikit-image/scikit-image/blob/master/skimage/color/colorconv.py
    """
    rgb=numpyArray(rgb)
    singleColor=len(rgb.shape)==1
    if singleColor: # convert to one-pixel image
        rgb=np.array([[rgb]])
    out = np.empty_like(rgb)
    # -- V channel
    out_v = rgb.max(-1)
    # -- S channel
    delta = rgb.ptp(-1)
    # Ignore warning for zero divided by zero
    old_settings = np.seterr(invalid='ignore')
    out_s = delta / out_v
    out_s[delta == 0.] = 0.
    # -- H channel
    # red is max
    idx = (rgb[:, :, 0] == out_v)
    out[idx, 0] = (rgb[idx, 1] - rgb[idx, 2]) / delta[idx]
    # green is max
    idx = (rgb[:, :, 1] == out_v)
    out[idx, 0] = 2. + (rgb[idx, 2] - rgb[idx, 0]) / delta[idx]
    # blue is max
    idx = (rgb[:, :, 2] == out_v)
    out[idx, 0] = 4. + (rgb[idx, 0] - rgb[idx, 1]) / delta[idx]
    out_h = (out[:, :, 0] / 6.) % 1.
    out_h[delta == 0.] = 0.
    np.seterr(**old_settings)
    # -- output
    out[:, :, 0] = out_h
    out[:, :, 1] = out_s
    out[:, :, 2] = out_v
    # remove NaN
    out[np.isnan(out)] = 0
    if singleColor:
        out=out[0,0]
    return out


def hsv2rgbImage(img):
    """
    same as rgb2hsvArray only returns an image

    :param img: image to convert.  can be pil image or numpy array

    :return img: returns an "rgb" image, that is actually "hsv"
    """
    return pilImage(hsv2rgbArray(img))


def hsv2rgbArray(hsv):
    """
    Transform an hsv image to an rgb array

    :param hsv: the input image.  can be pil image or numpy array
    :return rgb: the output array

    This comes from scikit-image:
        https://github.com/scikit-image/scikit-image/blob/master/skimage/color/colorconv.py
    """
    hsv=numpyArray(hsv)
    if len(hsv.shape)==2:
        # this is a black and white image, so treat it as value
        # with zero hue and saturation
        hs=np.empty((hsv.shape))
        return np.dstack((hs,hs,hsv))
    hi = np.floor(hsv[:, :, 0] * 6)
    f = hsv[:, :, 0] * 6 - hi
    p = hsv[:, :, 2] * (1 - hsv[:, :, 1])
    q = hsv[:, :, 2] * (1 - f * hsv[:, :, 1])
    t = hsv[:, :, 2] * (1 - (1 - f) * hsv[:, :, 1])
    v = hsv[:, :, 2]
    hi = np.dstack([hi, hi, hi]).astype(np.uint8) % 6
    out = np.choose(hi, [np.dstack((v, t, p)),
        np.dstack((q, v, p)),
        np.dstack((p, v, t)),
        np.dstack((p, q, v)),
        np.dstack((t, p, v)),
        np.dstack((v, p, q))])
    return out


def rgb2cmykArray(rgb):
    """
    Takes [[[r,g,b]]] colors in range 0..1
    Returns [[[c,m,y,k]]] in range 0..1
    """
    #k=rgb.sum(-1)
    c=1.0-rgb[:,:,0]
    m=1.0-rgb[:,:,1]
    y=1.0-rgb[:,:,2]
    minCMY=np.dstack((c,m,y)).min(-1)
    c=(c-minCMY)/(1.0-minCMY)
    m=(m-minCMY)/(1.0-minCMY)
    y=(y-minCMY)/(1.0-minCMY)
    k=minCMY
    return np.dstack((c,m,y,k))


def cmyk2rgbArray(cmyk):
    """
    Takes [[[c,m,y,k]]] colors in range 0..1
    Returns [[[r,g,b]]] in range 0..1
    """
    r=1-(min(1,cmyk[:,:,0]*(1-cmyk[:,:,3])+cmyk[:,:,3]))
    g=1-(min(1,cmyk[:,:,1]*(1-cmyk[:,:,3])+cmyk[:,:,3]))
    b=1-(min(1,cmyk[:,:,2]*(1-cmyk[:,:,3])+cmyk[:,:,3]))
    return np.dstack((r,g,b))


def rgb2cmykImage(img):
    """
    same as rgb2cmykArray only takes an image and returns an image

    :param img: image to convert

    :return img: returns an "rgba" image, that is actually "cmyk"
    """
    rgb=numpyArray(img)
    final=rgb2cmykArray(rgb)
    return pilImage(final)


def getChannel(img,channel):
    """
    get a channel as a new image.

    :param img: the image to extract a color channel from
    :param channel: name of the channel to extract - supports R,G,B,A,H,S,V

    Returns a grayscale image, or None
    """
    img=numpyArray(img)
    if channel=='R':
        if len(img.shape)==2:
            img=img[:,:]
        else:
            img=img[:,:,0]
    elif channel=='G':
        if len(img.shape)==2:
            img=img[:,:]
        else:
            img=img[:,:,1]
    elif channel=='B':
        if len(img.shape)==2:
            img=img[:,:]
        else:
            img=img[:,:,2]
    elif channel=='A':
        if len(img.shape)==2 or len(img[0,0])<4:
            img=np.ones(img.shape)
        else:
            img=img[:,:,3]
    elif channel=='H':
        if len(img.shape)==2:
            img=np.zeros(img.shape)
        else:
            img=rgb2hsvArray(img)[:,:,0]
    elif channel=='S':
        if len(img.shape)==2:
            img=np.zeros(img.shape)
        else:
            img=rgb2hsvArray(img)[:,:,1]
    elif channel=='V':
        if len(img.shape)==2:
            pass
        else:
            img=rgb2hsvArray(img)[:,:,2]
    elif channel=='C':
        if len(img.shape)==2:
            img=np.zeros(img.shape)
        else:
            img=rgb2cmykArray(img)[:,:,0]
    elif channel=='M':
        if len(img.shape)==2:
            img=np.zeros(img.shape)
        else:
            img=rgb2cmykArray(img)[:,:,1]
    elif channel=='Y':
        if len(img.shape)==2:
            img=np.zeros(img.shape)
        else:
            img=rgb2cmykArray(img)[:,:,2]
    elif channel=='K':
        if len(img.shape)==2:
            pass
        else:
            img=rgb2cmykArray(img)[:,:,3]
    else:
        raise Exception('Unknown channel: '+channel)
    return img


def grayscale(img,method='BT.709'):
    """
    Convert any image to grayscale

    :param method: available are:
        "max" - max of all channels
        "min" - min of all channels
        "avg" - average of all channels
        "HSV" or "median" - average of max and min
        "ps-gimp" - weighted average used by most image applications
        "BT.709" - [default] weighted average specified by
            ITU-R "luma" specification BT.709
        "BT.601" - weighted average specified by ITU-R specification BT.601
            used by video formats

    NOTE:
        There are many different ways to go about this.
        http://www.tannerhelland.com/3643/grayscale-image-algorithm-vb6/
    """
    img=numpyArray(img)
    if len(img.shape)>2:
        if method=='max':
            img=img.max(-1)
        elif method=='min':
            img=img.min(-1)
        elif method=='avg':
            img=img.sum(-1)/3.0
        elif method=='ps-gimp':
            img=(img[:,:,0]*0.3 + img[:,:,1]*0.59 + img[:,:,2]*0.11)
        elif method=='BT.709':
            img=(img[:,:,0]*0.2126 + img[:,:,1]*0.7152 + img[:,:,2]*0.0722)
        elif method=='BT.601':
            img=(img[:,:,0]*0.299 + img[:,:,1]*0.587 + img[:,:,2]*0.114)
        elif method in ('HSV','median'):
            img=(img.max(-1)+img.min(-1))/2.0
    return img


def cmdline(args):
    """
    Run the command line

    :param args: command line arguments (WITHOUT the filename)
    """
    printHelp=False
    if not args:
        printHelp=True
    else:
        img=None
        from .helper_routines import defaultLoader
        for arg in args:
            if arg.startswith('-'):
                arg=[a.strip() for a in arg.split('=',1)]
                if arg[0] in ['-h','--help']:
                    printHelp=True
                elif arg[0]=='--channel':
                    ch=getChannel(img,arg[1])
                    if ch is None:
                        print('No channel:',arg[1])
                    else:
                        ch.show()
                else:
                    print('ERR: unknown argument "'+arg[0]+'"')
            else:
                img=defaultLoader(arg)
    if printHelp:
        print('Usage:')
        print('  colorSpaces.py [options]')
        print('Options:')
        print('   --channel=R .............. extract a channel from the image - R,G,B,A,H,S,V') # noqa: E501 # pylint: disable=line-too-long
        print('Notes:')
        print('   * All filenames can also take file:// http:// https:// ftp:// urls') # noqa: E501 # pylint: disable=line-too-long


if __name__=='__main__':
    import sys
    cmdline(sys.argv[1:])

#!/usr/bin/env
# -*- coding: utf-8 -*-
"""
This allows easy conversion between
    pil images
    numpy arrays
    filenames
    int/float
all of different numbers of color channels
"""
import typing
import os
try:
    # first try to use bohrium, since it could help us accelerate
    # https://bohrium.readthedocs.io/users/python/
    import bohrium as np # type: ignore
except ImportError:
    # if not, plain old numpy is good enough
    import numpy as np
from PIL import Image
from . import resizing


#-------------- numpy <-> PIL smart conversion

def defaultLoader(f:typing.Union[str,typing.IO])->Image.Image:
    """
    load an image from a file-like object, filename, or url of type
        file://
        ftp://
        sftp://
        http://
        https://
    """
    if isinstance(f,(np.ndarray,Image.Image)):
        return f
    if f is None:
        return Image.new('RGBA',(1,1),(255,255,255,0))
    if isinstance(f,str):
        proto=f.split('://',1)
        if len(proto)>2 and proto.find('/')<0:
            if proto[0]=='file':
                f=proto[-1]
                if os.sep!='/':
                    f=f.replace('/',os.sep)
                    if f[1]=='|':
                        f[1]=':'
            else:
                proto=proto[1]
        else:
            proto='file'
        if proto!='file':
            import urllib.request
            import urllib.error
            import urllib.parse
            import io
            # some servers only like "real browsers"
            headers={'User-Agent':'Mozilla 5.10'}
            request=urllib.request.Request(f,None,headers)
            response=urllib.request.urlopen(request)
            f=io.StringIO(response.read().decode('utf-8'))
    return Image.open(f)

def numpyArray(img,floatingPoint:bool=True,loader=None)->np.ndarray:
    """
    always return a writeable ndarray

    if img is a pil image, convert it.
    if it's already an array, return it

    :param img: can be a pil image, a numpy array, or anything loader can load
    :param floatingPoint: return a float array vs return a byte array
    :param loader: return a tool used to load images from strings.
        if None, use defaultLoader() in this file
    """
    if isinstance(img,str) or hasattr(img,'read'):
        if loader is None:
            loader=defaultLoader
        img=loader(img)
    if isinstance(img,np.ndarray):
        a=img
    else:
        a=np.asarray(img)
    if not a.flags.writeable:
        a=a.copy()
    if a is not None and isFloat(a)!=floatingPoint:
        if floatingPoint:
            a=a/255.0
        else:
            a=np.int(a*255)
    return a


def pilImage(img,loader=None)->Image:
    """
    converts anything to a pil image

    :param img: can be a pil image, loadable file path, or a numpy array
    """
    if isinstance(img,Image.Image):
        # already what we need
        pass
    elif isinstance(img,str) or hasattr(img,'read'):
        # load it with the loader
        if loader is None:
            loader=defaultLoader
        img=loader(img)
    else:
        # convert numpy array
        from .helper_routines import clampImage
        img=clampImage(img)
        mode=imageMode(img)
        if isFloat(img):
            img=np.round(img*255)
        img=Image.fromarray(img.astype('uint8'),mode)
    return img


#-------------- color mode/info

def imageMode(img)->str:
    """
    :param img: can be:
      a textual image mode, numpy array, or an image

    :return bool:
    """
    if isinstance(img,str):
        return img
    if isinstance(img,np.ndarray):
        if len(img.shape)<3: # black and white is [x,y,val] not [x,y,[val]]
            return 'L'
        modeGuess=['L','LA','RGB','RGBA']
        return modeGuess[img.shape[2]-1]
    return img.mode


def isFloat(img)->bool:
    """
    Decide whether image pixels or an individual color
    is floating point or byte

    :param img: can either one be a numpy array, or an image

    :return bool:
    """
    if isinstance(img,np.ndarray):
        if len(img.shape)<2: # a single color
            if img.shape or isinstance(img[0],(np.float,np.float64)):
                return True
        elif len(img.shape)<3: # black and white is [x,y,val] not [x,y,[val]]
            if isinstance(img[0,0],(np.float,np.float64)):
                return True
        else:
            if isinstance(img[0,0,0],(np.float,np.float64)):
                return True
    if isinstance(img,(tuple,list)):
        # a single color
        if isinstance(img[0],float):
            return True
    if isinstance(img,float):
        return True
    return False


def hasAlpha(mode)->bool:
    """
    determine if an image mode is has an alpha channel

    :param mode: can either a textual image mode, numpy array, or an image
    :return bool:
    """
    if not isinstance(mode,str):
        mode=imageMode(mode)
    return mode[-1]=='A'


def getAlpha(image,alwaysCreate:bool=True):
    """
    gets the alpha channel regardless of image type

    :param image: the image whose mask to get
    :param alwaysCreate: always returns a numpy array
        (otherwise, may return None)

    :return: alpha channel as a PIL image, or numpy array,
        or possibly None, depending on alwaysCreate
    """
    ret=None
    if image is None or not hasAlpha(image):
        if alwaysCreate:
            ret=np.array(image.size())
            ret.fill(1.0)
    elif isinstance(image,Image.Image):
        ret=image.getalpha()
    else:
        ret=image[:,:,-1]
    return ret


def setAlpha(image,alpha):
    """
    sets the alpha mask regardless of image type

    :param image: the image to be changed
    :param mask: a mask image to use to cut out the image it can be:
        image with alpha channel to steal
        -or-
        grayscale (white=opaque, black=transparent)

    :returns: adjusted image (could be PIL image or numpy array, depending on
        what's expedient.  If you need a particular one, wrap the call in
        pilImage() or numpyArray())

    NOTE: if alpha channel exists, will be darkened such that
        a hole in either mask results in a hole

    IMPORTANT: the image bits may be altered.  To prevent this,
        set image.immutable=True
    """
    if image is None or alpha is None:
        return image
    if isinstance(image,Image.Image):
        # make sure not to smash any bits we're keeping
        if hasattr(image,'immutable') and image.immutable:
            image=image.copy()
    if imageMode(alpha)!='L':
        # make sure we have a grayscale to combine
        alpha=getAlpha(alpha,alwaysCreate=False)
        if alpha is None:
            return image
    image,alpha=resizing.makeSameSize(image,alpha,(0,0,0,0))
    if hasAlpha(image):
        channels=np.asarray(image)
        alpha1=np.minimum(channels[:,:,-1],alpha) # Darken blend mode
        image=np.dstack((channels[:,:,0:-1],alpha1))
    else:
        if isinstance(image,Image.Image):
            if not isinstance(alpha,Image.Image):
                alpha=pilImage(alpha)
            image.putalpha(alpha)
        else:
            if isinstance(alpha,Image.Image):
                alpha=np.asarray(alpha)
            image=np.dstack((channels[:,:,0:-1],alpha1))
    return image


def isColor(mode)->bool:
    """
    determine if an image mode is color (or grayscale)

    :param mode: can either be a textual image mode, numpy array, or an image
    :return bool:
    """
    if not isinstance(mode,str):
        mode=imageMode(mode)
    return mode[0]!='L'


def maxMode(mode1,mode2='L',requireAlpha=False):
    """
    Finds the maximum color mode.

    :param mode1: can either be a textual image mode, numpy array, or an image
    :param mode2: can either be a textual image mode, numpy array, or an image
    :param requireAlpha: stipulate that the result must have an alpha channel

    :return mode: best image mode
    """
    if isColor(mode1) or isColor(mode2):
        ret='RGB'
    else:
        ret='L'
    if requireAlpha or hasAlpha(mode1) or hasAlpha(mode2):
        ret=ret+'A'
    return ret


def changeMode(img,mode):
    """
    changes the image to a different color mode
    """
    currentMode=imageMode(img)
    if currentMode!=mode:
        if isinstance(img,np.ndarray):
            # TODO: this would be faster to do in array-land, but I'm too lazy
            #\tmess with the if/then spaghetti required for that.
            img=pilImage(img)
            img=img.convert(mode)
        else:
            img=img.convert(mode)
    return img


def makeSameMode(images):
    """
    takes an array of images, returns an array of images
    converts all into the max image mode of all images in the list
    """
    maxmode='L'
    modeDifferences=False
    for i,img in enumerate(images):
        if i==0:
            maxmode=imageMode(img)
        else:
            mode=imageMode(img)
            if mode!=maxmode:
                modeDifferences=True
                maxmode=maxMode(maxmode,mode)
    if modeDifferences:
        images=[changeMode(img,maxmode) for img in images]
    return images


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
        print('  imageRepr.py [options]')
        print('Options:')
        print('   NONE')


if __name__=='__main__':
    import sys
    cmdline(sys.argv[1:])

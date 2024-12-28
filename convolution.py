#!/usr/bin/env
# -*- coding: utf-8 -*-
"""
Contains routines to convolve an image
"""
import typing
try:
    # first try to use bohrium, since it could help us accelerate
    # https://bohrium.readthedocs.io/users/python/
    import bohrium as np # type: ignore
except ImportError:
    # if not, plain old numpy is good enough
    import numpy as np
import scipy # type: ignore
from scipy import ndimage

from helper_routines import highlights
from .imageRepr import numpyArray


CONVOLVE_MATTRICES={
    'emboss':[[-2,-1, 0],
              [-1, 1, 1],
              [ 0, 1, 2]],
    'edgeDetect':[[ 0, 1, 0],
                  [ 1,-4, 1],
                  [ 0, 1, 0]],
    'edgeEnhance':[[ 0, 0, 0],
                   [-1, 1, 0],
                   [ 0, 0, 0]],
    'blur':[[ 0, 0, 0, 0, 0],
            [ 0, 1, 1, 1, 0],
            [ 0, 1, 1, 1, 0],
            [ 0, 1, 1, 1, 0],
            [ 0, 0, 0, 0, 0]],
    'sharpen':[[ 0, 0, 0, 0, 0],
               [ 0, 0,-1, 0, 0],
               [ 0,-1, 5,-1, 0],
               [ 0, 0,-1, 0, 0],
               [ 0, 0, 0, 0, 0]],
    'xblur':[[0,0,0],
             [1,1,1],
             [0,0,0]],
    'yblur':[[0,1,0],
             [0,1,0],
             [0,1,0]],
    'sobelX':[[1,2,1],
             [0,0,0],
             [-1,-2,-1]],
    'sobelY':[[1,0,-1],
             [2,0,-2],
             [1,0,-1]],
}


def directionalBlur(img,size=15):
    """
    blur in a given direction

    https://www.packtpub.com/mapt/book/application_development/9781785283932/2/ch02lvl1sec21/motion-blur#!
    """
    oSize=img.shape
    kernel= np.zeros((size,size))
    kernel[int((size-1)/2), :] = np.ones(size)
    kernel= kernel / size
    img=scipy.signal.convolve(img,kernel)
    d=(img.shape[0]-oSize[0])/2,(img.shape[1]-oSize[1])/2
    img=img[d[0]:-d[0],d[1]:-d[1]]
    return img


def circularBlur(img):
    """
    blur in a radial manner

    http://chemaguerra.com/circular-radial-blur/
    """
    from . import numberSpaces
    polar=numberSpaces.cartesian2polar(img)
    polar=convolve(polar,'xblur')
    return numberSpaces.polar2cartesian(polar)


def zoomBlur(img):
    """
    blur in a zoom manner

    http://chemaguerra.com/circular-radial-blur/
    """
    from . import numberSpaces
    polar=numberSpaces.cartesian2polar(img)
    polar=convolve(polar,'yblur')
    return numberSpaces.polar2cartesian(polar)


def streak(img,size=15):
    """
    The idea is to cause streaks in a given direction.

    TODO: not working
    """
    return img+directionalBlur(highlights(img,.95),size)


def convolve(
    img,
    matrix,
    add:float=0,
    divide:float=1,
    edge:str='clamp'):
    """
    run a given convolution matrix

    :param matrix: can be a numerical matrix or an entry in CONVOLVE_MATTRICES
    """
    if isinstance(matrix,str):
        if matrix.find(',')>0:
            matrix=''.join(matrix.split())
            if matrix.startswith('[['):
                matrix=matrix[1:-1]
            matrix=[
                [float(col) for col in row.split(',')]
                for row in matrix[1:-1].replace('],','').split('[')]
        else:
            matrix=CONVOLVE_MATTRICES[matrix]
    size=len(matrix)
    border=size/2-1
    #img=imageBorder(img,border,edge)
    #k=ImageFilter.Kernel((size,size),matrix,scale=divide,offset=add)
    #img=img.filter(k)
    img=numpyArray(img)
    if len(img.shape)>2:
        ret=[]
        for ch in range(img.shape[2]):
            ret.append(ndimage.convolve(img[:,:,ch],matrix))
        img=np.dstack(ret)
    else:
        img=ndimage.convolve(img,matrix)
    return img


def cmdline(args:typing.Iterable[str])->int:
    """
    Run the command line

    :param args: command line arguments (WITHOUT the filename)
    """
    printhelp=False
    if not args:
        printhelp=True
    else:
        from .imageRepr import pilImage
        img=None
        for arg in args:
            if arg.startswith('-'):
                arg=[a.strip() for a in arg.split('=',1)]
                if arg[0] in ['-h','--help']:
                    printhelp=True
                elif arg[0]=='--save':
                    pilImage(img).save(arg[1])
                elif arg[0]=='--show':
                    pilImage(img).show()
                elif arg[0]=='--convolve':
                    img=convolve(img,arg[1])
                elif arg[0]=='--circularBlur':
                    img=circularBlur(img)
                elif arg[0]=='--zoomBlur':
                    img=zoomBlur(img)
                else:
                    print('ERR: unknown argument "'+arg[0]+'"')
            else:
                img=arg
    if printhelp:
        print('Usage:')
        print('  convolution.py image.jpg [options]')
        print('Options:')
        print('   --convolve=matrix ..... matrix can be a name or a matrix')
        print('                           of numbers')
        print('   --circularBlur=amount . blur around circular center point')
        print('   --zoomBlur=amount ..... blur outward from a circular')
        print('                           center point')
        print('   --save=filename ....... save the current image')
        print('   --show ................ show the current image')
        return -1
    return 0


if __name__=='__main__':
    import sys
    cmdline(sys.argv[1:])

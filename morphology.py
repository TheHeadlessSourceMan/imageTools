#!/usr/bin/env
# -*- coding: utf-8 -*-
"""
Contains mathematical morphology routines (dilate, erode, etc)

See also:
    https://en.wikipedia.org/wiki/Mathematical_morphology
"""
import typing
import scipy # type: ignore
from .imageRepr import numpyArray,Image
from .helper_routines import applyFunctionToPatch


def erode(img,size:int=3)->Image:
    """
    implement image erosion

    """
    img=numpyArray(img)
    def erodeFn(p):
        p[1,1]=1
        return p
    applyFunctionToPatch(erodeFn,img,(size,size))
    return Image.fromarray(img.astype('uint8'),img.mode)


def dilate(img,size:int=3)->Image:
    """
    implement image dilation
    """
    img=numpyArray(img)
    img=scipy.ndimage.grey_dilation(img,size=(size,size))
    return Image.fromarray(img)


def cmdline(args:typing.Iterable[str])->int:
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
        print('  morphology.py [options]')
        print('Options:')
        print('   NONE')
        return -1
    return 0


if __name__=='__main__':
    import sys
    cmdline(sys.argv[1:])

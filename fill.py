#!/usr/bin/env
# -*- coding: utf-8 -*-
"""
Fill regions of an image
"""


from .selectionsAndPaths import *


def fill(img,color,location=None,tolerance=0,soften=10,smartSoften=True,colorspace='RGB',pickMode='average',preserveAlpha=False):
    """
    A general drawing program fill tool.

    This works by using the select tool, then changing the colors
    of selected pixels.  Create a layer underneath it with the target color, and then merge the two together.

    :param img: can be a pil image, numpy array, etc
    :param color: the color to fill with
    :param location: the point at which to perform the fill.  If None, fill the entire image. can be a single point or an [x,y,w,h]
    :param tolerance: how close the selection must be
    :param soften: apply a soften radius to the edges of the selection (soften will result in the color being blended with the original)
        TODO: does this mean we need a blend mode?
    :param smartSoften: multiply the soften radius by how near the pixel is to the selection color
    :param colorspace: change the given img (assumed to be RGB) into another corlorspace before matching
        TODO: implement this!
    :param preserveAlpha: Interesting tool to preserve any transparency on the image
    """
    selection=selectByPoint(img,location,tolerance=0,soften=10,smartSoften=True,colorspace='RGB',pickMode='average')
    raise NotImplementedError()


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
        print('  fill.py [options]')
        print('Options:')
        print('   NONE')


if __name__=='__main__':
    import sys
    cmdline(sys.argv[1:])

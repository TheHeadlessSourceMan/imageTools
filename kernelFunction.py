#!/usr/bin/env
# -*- coding: utf-8 -*-
"""
This allows you to apply a kernel+function to all pixels in the image
"""

class KernelMath:
    """
    feeds a math operation a kernel matrix and set of pixels

    used for as general-purpose image math tool

    TODO: if it works, and we can get it on GPU, this could replace or augment many of
        the imgTools, for instance convolution and morphology
    """

    CONVOLUTION="sum(px[x,y]*k[x,y])" # TODO: something like this? (plus other two rows)
    THRESHOLD_MASK="px[x,y]>k[x,y]" # takes a kernel of size 1
    THRESHOLD="px[x,y]>k[x,y]?px[x,y]" # takes a kernel of size 1
    LOCAL_AVERAGE="sum(px[x,y])/(w*h)"
    SOBEL2D="sum(px[x,y]*k[x,y])+sum(px[x,y]*k[y,x])" # Normal sobel x or y is just a convolution, but this does both directions (pass in kernel [[1,2,1],[0,0,0],[-1,-2,-1]] )
    ERODE="and(px[x,y]>k[x,y])" # kernel values are used as threshold
    DILATE="or(px[x,y]>k[k,y])" # kernel values are used as threshold
    EDGE_MORPHO="(sum(px[x,y])-px[x,y])/px[x,y]" # morphology-based edge detection

    def __init__(self,kernel,math=None):
        """
        :param kernel: a 2d matrix of coeffiecents to send into the math along with the pixels
        :param math: the mathematical function to apply to the kernel and pixels
            the varaibles px and k are the pixels and kernel
            x,y are used across all indices (this is positive/negative such that 0,0 is the pixel under operation)
            w,h are integers for the max x and y
            NOTE: if not specified, use convolution math
        """
        if math is None:
            math=self.CONVOLUTION_MATH
        self.kernel=kernel
        self.math=math


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
                    print('ERR: unknown argument "'+arg[0]+'"')
            else:
                print('ERR: unknown argument "'+arg+'"')
    if printhelp:
        print('Usage:')
        print('  kernelFunction.py [options]')
        print('Options:')
        print('   NONE')


if __name__=='__main__':
    import sys
    cmdline(sys.argv[1:])

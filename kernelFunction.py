#!/usr/bin/env
# -*- coding: utf-8 -*-
"""
This allows you to apply a kernel+function to all pixels in the image
"""

class KernelMath:
    """
    feeds a math operation a kernel matrix and set of pixels

    used for as general-purpose image math tool

    TODO: if it works, and we can get it on GPU,
        this could replace or augment many of
        the imgTools, for instance convolution and morphology
    """

    # TODO: something like this? (plus other two rows)
    CONVOLUTION="sum(px[x,y]*k[x,y])"

    # takes a kernel of size 1
    THRESHOLD_MASK="px[x,y]>k[x,y]"

    # takes a kernel of size 1
    THRESHOLD="px[x,y]>k[x,y]?px[x,y]"

    LOCAL_AVERAGE="sum(px[x,y])/(w*h)"

    # Normal sobel x or y is just a convolution, but this does both directions
    # (pass in kernel [[1,2,1],[0,0,0],[-1,-2,-1]] )
    SOBEL2D="sum(px[x,y]*k[x,y])+sum(px[x,y]*k[y,x])"

    # kernel values are used as threshold
    ERODE="and(px[x,y]>k[x,y])"

    # kernel values are used as threshold
    DILATE="or(px[x,y]>k[k,y])"

    # morphology-based edge detection
    EDGE_MORPHO="(sum(px[x,y])-px[x,y])/px[x,y]"

    def __init__(self,kernel,math=None):
        """
        :param kernel: a 2d matrix of coefficients to send into the math
            along with the pixels
        :param math: the mathematical function to apply to
            the kernel and pixels
            the variables px and k are the pixels and kernel
            x,y are used across all indices
                (this is positive/negative such that 0,0
                is the pixel under operation)
            w,h are integers for the max x and y
            NOTE: if not specified, use convolution math
        """
        if math is None:
            math=self.CONVOLUTION
        self.kernel=kernel
        self.math=math


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
        print('  kernelFunction.py [options]')
        print('Options:')
        print('   NONE')


if __name__=='__main__':
    import sys
    cmdline(sys.argv[1:])

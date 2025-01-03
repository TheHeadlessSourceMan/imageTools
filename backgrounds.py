#!/usr/bin/env
# -*- coding: utf-8 -*-
"""
Routines to create fill patterns using PIL
"""
import typing
from PIL import Image, ImageOps


def checkerboard(
    w:float,h:float,
    color1=(102,102,102),color2=(153,153,153),
    squareSize:int=8
    )->Image.Image:
    """
    create an image based on checkerboard of two colors

    default is a "transparent" checkerboard background image ala image editors
    """
    l1,l2=len(color1),len(color2)
    if l1!=l2:
        if l1<l2:
            c=[]
            c.extend(color1)
            for _ in range(len(l2-l1)):
                c.append('255')
            color1=(ch for ch in c)
            l1=l2
        else:
            c=[]
            c.extend(color2)
            for _ in range(l1-l2):
                c.append('255')
            color2=(ch for ch in c)
            l2=l1
    if l1==0:
        mode="L"
    elif l1==3:
        mode="RGB"
    elif l1==4:
        mode="RGBA"
    else:
        raise Exception("Unable to determine mode from colors")
    img=Image.new(mode,(int(w),int(h)),color1)
    square=Image.new(mode,(int(squareSize),int(squareSize)),color2)
    even=True
    for x in range(0,w,squareSize):
        if even:
            for y in range(0,h,squareSize*2):
                img.paste(square,(int(x),int(y)))
            even=False
        else:
            for y in range(squareSize,h-squareSize,squareSize*2):
                img.paste(square,(int(x),int(y)))
            even=True
    return img


def blinds(
    w:float,h:float,
    orientation='horizontal',
    color1=(0,0,0,255),color2=(0,0,0,0),
    thickness=1
    )->Image.Image:
    """
    create an image based on lines of two colors

    default is 1pixel black lines, transparent background
    """
    if orientation=='vertical':
        img=blinds(h,w,orientation='horizontal',
            color1=color1,color2=color2,thickness=thickness)
        img.rotate(90)
    else:
        img=Image.new("RGBA",(int(w),int(h)),color1)
        square=Image.new('RGBA',(img.width,int(thickness)),color2)
        for y in range(0,h,thickness*2):
            img.paste(square,(0,int(y)))
    return img


def tile1d(
    w:float,
    tileImg:Image.Image,
    mode:str='repeat'
    )->Image.Image:
    """
    mode: 'repeat' 'once' 'stretch' 'repeat_flip'
    """
    if mode=='repeat':
        img=Image.new(tileImg.mode,(int(w),tileImg.height))
        for x in range(0,w,tileImg.width):
            img.paste(tileImg,(int(x),0))
    elif mode=='repeat_flip':
        img=Image.new(tileImg.mode,(int(w),tileImg.height))
        mirrored=ImageOps.mirror(tileImg)
        even=True
        y=0
        for x in range(0,w,tileImg.width):
            if even:
                img.paste(tileImg,(int(x),int(y)))
                even=False
            else:
                img.paste(mirrored,(int(x),int(y)))
                even=True
    elif mode=='once':
        img=Image.new(tileImg.mode,(int(w),tileImg.height))
        img.paste(tileImg,(0,0))
    elif mode=='stretch':
        img=tileImg.copy()
        img.resize(
            (int(w),tileImg.height),
            Image.ANTIALIAS) # pylint: disable=no-member
    else:
        raise Exception('ERR: unknown image repeat mode "'+mode+'"')
    return img


def tile(
    w:float,h:float,
    tileImg:Image.Image,
    xMode:str='repeat',yMode:str='repeat'
    )->Image.Image:
    """
    xMode: 'repeat' 'once' 'stretch' 'repeat_flip'
    yMode: 'repeat' 'once' 'stretch' 'repeat_flip'
    """
    tileImg.rotate(90)
    tileImg=tile1d(h,tileImg,mode=yMode)
    tileImg.rotate(-90)
    tileImg=tile1d(w,tileImg,mode=xMode)
    return tileImg


def cmdline(args:typing.Iterable[str])->int:
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
        print('  backgrounds.py [options]')
        print('Options:')
        print('   NONE')


if __name__=='__main__':
    import sys
    cmdline(sys.argv[1:])

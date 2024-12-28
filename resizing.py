#!/usr/bin/env
# -*- coding: utf-8 -*-
"""
Contains a litany of resizing/pasting/cropping routines
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
from PIL import Image, ImageOps
from . import imageRepr
from .bounds import Bounds


def imageBorder(img,thickness,edgeFill="#ffffff00"):
    """
    Add a border of thickness pixels around the image

    :param img: the image to add a border to can be pil image,
        numpy array, or whatever
    :param thickness: the border thickness in pixels.  Can be:
        int - border all the way around
        (w_border,h_border) - add border this big to each side
        (x,y,x2,y2) - add border this big to each side
    :param edgeFill: defines how to extend.  It can be:
        mirror - reflect the pixels leading up to the border
        repeat - repeat the image over again (useful with repeating textures)
        clamp - streak last pixels out to edge
        [background color] - simply fill with the given color

    TODO: combine into extendImageCanvas function
    """
    if not isinstance(thickness,(tuple,list)):
        thickness=(thickness,thickness,thickness,thickness)
    elif len(thickness)==2:
        thickness=(thickness[0],thickness[1],thickness[0],thickness[1])
    thickness=[int(t) for t in thickness]
    img=imageRepr.pilImage(img)
    newSize=(
        int(img.size[0]+thickness[0]+thickness[2]),
        int(img.size[1]+thickness[1]+thickness[3]))
    if edgeFill=='mirror':
        newImage=Image.new(img.mode,newSize)
        # top
        fill=ImageOps.flip(img.crop(
            (0,0,img.width,thickness[1])))
        newImage.paste(fill,(thickness[1],0))
        # bottom
        fill=ImageOps.flip(img.crop(
            (0,img.height-thickness[2],img.width,img.height)))
        newImage.paste(fill,(thickness[2],img.height+thickness[2]))
        # left
        fill=ImageOps.mirror(img.crop(
            (0,0,thickness[0],img.height)))
        newImage.paste(fill,(0,thickness[0]))
        # right
        fill=ImageOps.mirror(img.crop(
            (img.width-thickness[3],0,img.width,img.height)))
        newImage.paste(fill,(img.width+thickness[3],thickness[3]))
        # top-left corner
        #fill=ImageOps.mirror(ImageOps.flip(img.crop(
        #   (0,0,thickness,thickness))))
        #newImage.paste(fill,(0,0))
        # top-right corner
        #fill=ImageOps.mirror(ImageOps.flip(img.crop(
        #   (img.width-thickness,0,img.width,thickness))))
        #newImage.paste(fill,(img.width+thickness,0))
        # bottom-left corner
        #fill=ImageOps.mirror(ImageOps.flip(img.crop(
        #   (0,img.height-thickness,thickness,img.height))))
        #newImage.paste(fill,(0,img.height+thickness))
        # bottom-right corner
        #fill=ImageOps.mirror(ImageOps.flip(img.crop(
        #   (img.width-thickness,img.height-thickness,img.width,img.height))))
        #newImage.paste(fill,(img.width+thickness,img.height+thickness))
    elif edgeFill=='repeat':
        newImage=Image.new(img.mode,newSize)
        # top
        fill=img.crop((0,0,img.width,thickness[1]))
        newImage.paste(fill,(thickness[1],img.height+thickness[1]))
        # bottom
        fill=img.crop((0,img.height-thickness[2],img.width,img.height))
        newImage.paste(fill,(thickness[2],0))
        # left
        fill=img.crop((0,0,thickness[0],img.height))
        newImage.paste(fill,(img.width+thickness[0],thickness[0]))
        # right
        fill=img.crop((img.width-thickness[3],0,img.width,img.height))
        newImage.paste(fill,(0,thickness[3]))
        # top-left corner
        fill=img.crop((0,0,thickness,thickness))
        newImage.paste(fill,(img.width+thickness,img.height+thickness))
        # top-right corner
        fill=img.crop((img.width-thickness,0,img.width,thickness))
        newImage.paste(fill,(0,img.height+thickness))
        # bottom-left corner
        fill=img.crop((0,img.height-thickness,thickness,img.height))
        newImage.paste(fill,(img.width+thickness,0))
        # bottom-right corner
        fill=img.crop((
            img.width-thickness,
            img.height-thickness,
            img.width,
            img.height))
        newImage.paste(fill,(0,0))
    elif edgeFill=='clamp':
        newImage=Image.new(img.mode,newSize)
        # top
        fill=img.crop((0,0,img.width,1)).resize(
            (img.width,thickness[1]),
            resample=Image.NEAREST) # pylint: disable=no-member
        newImage.paste(fill,(thickness[1],0))
        # bottom
        fill=img.crop((0,img.height-1,img.width,img.height)).resize(
            (img.width,thickness[2]),
            resample=Image.NEAREST) # pylint: disable=no-member
        newImage.paste(fill,(thickness[2],img.height+thickness[2]))
        # left
        fill=img.crop((0,0,1,img.height)).resize(
            (thickness[0],img.height),
            resample=Image.NEAREST) # pylint: disable=no-member
        newImage.paste(fill,(0,thickness[0]))
        # right
        fill=img.crop((img.width-1,0,img.width,img.height)).resize(
            (thickness[3],img.height),
            resample=Image.NEAREST) # pylint: disable=no-member
        newImage.paste(fill,(img.width+thickness[3],thickness[3]))
        # TODO: corners
        # top-left corner
        fill=img.crop((0,0,1,1)).resize(
            (thickness,thickness),
            resample=Image.NEAREST) # pylint: disable=no-member
        newImage.paste(fill,(0,0))
        # top-right corner
        fill=img.crop((img.width-1,0,img.width,1)).resize(
            (thickness,thickness),
            resample=Image.NEAREST) # pylint: disable=no-member
        newImage.paste(fill,(img.width+thickness,0))
        # bottom-left corner
        fill=img.crop((0,img.height-1,1,img.height)).resize(
            (thickness,thickness),
            resample=Image.NEAREST) # pylint: disable=no-member
        newImage.paste(fill,(0,img.height+thickness))
        # bottom-right corner
        fill=img.crop((img.width-1,img.height-1,img.width,img.height)).resize(
            (thickness,thickness),
            resample=Image.NEAREST) # pylint: disable=no-member
        newImage.paste(fill,(img.width+thickness,img.height+thickness))
    else:
        newImage=Image.new(img.mode,newSize,edgeFill)
    # splat the original image in the middle
    if True:
        if newImage.mode.endswith('A'):
            newImage.alpha_composite(img,dest=(thickness[0],thickness[1]))
        else:
            newImage.paste(img,(thickness[0],thickness[1]))
    return newImage


def extendImageCanvas(pilImage,newBounds,edgeFill=(128,128,128,0)):
    """
    Make pilImage the correct canvas size/location.

    :param pilImage: the image to move
    :param newBounds: - (w,h) or (x,y,w,h) or Bounds object
    :param edgeFill: color to use when extending the canvas
        (automatically choses image mode based on this color)

    :return new PIL image:

    NOTE: always creates a new image, so no original bits are altered
    """
    if isinstance(newBounds,tuple):
        if len(newBounds)>2:
            x,y,w,h=newBounds
        else:
            x,y=0,0
            w,h=newBounds
    else:
        x,y,w,h=newBounds.x,newBounds.y,newBounds.w,newBounds.h
    if w<pilImage.width or h<pilImage.height:
        fromSize=f'({pilImage.width},{pilImage.height})'
        toSize=f'({w},{h})'
        msg=f'Cannot "extend" canvas to smaller size. {fromSize} to {toSize}'
        raise ValueError(msg)
    if not isinstance(edgeFill,(list,tuple)) or len(edgeFill)<2:
        mode="L"
    elif len(edgeFill)<4:
        mode="RGB"
    else:
        mode="RGBA"
    mode=imageRepr.maxMode(mode,pilImage)
    img=Image.new(mode,(int(w),int(h)),edgeFill)
    x=max(0,w/2-pilImage.width/2)
    y=max(0,h/2-pilImage.height/2)
    if imageRepr.hasAlpha(mode):
        if not imageRepr.hasAlpha(pilImage):
            pilImage=pilImage.convert(
                imageRepr.maxMode(pilImage,requireAlpha=True))
        img.alpha_composite(pilImage,dest=(int(x),int(y)))
    else:
        img.paste(pilImage,box=(int(x),int(y)))
    return img


def paste(
    image,
    overImage,
    position:typing.Tuple[float,float]=(0,0),
    resize:bool=True):
    """
    A simple, dumb, paste operation like PIL's paste,
    only automatically uses alpha

    :param image: the image to be pasted on top (if None, returns overImage)
    :param overImage: the image will be pasted over the top of this image
        (if None, returns image)
    :param position: the position to place the new image,
        relative to overImage (can be negative)
    :param resize: allow the resulting image to be resized if overImage
        extends beyond its bounds
        TODO: need to implement this

    :returns: a combined image, or None if both image and overImage are None

    NOTE: this is effectively the same as doing
        blend(image,'normal',overImage,position,resize)

    IMPORTANT: the image bits may be altered.
        To prevent this, set image.immutable=True
    """
    image=imageRepr.pilImage(image)
    if image is None:
        return overImage
    if overImage is None:
        if position==(0,0): # no change
            return image
        # create a blank background
        overImage=Image.new(
            imageRepr.maxMode(image,requireAlpha=True),
            (int(image.width+position[0]),
            int(image.height+position[1])))
    elif (position[0]<0) \
        or (position[1]<0) \
        or (image.width+position[0]>overImage.width) \
        or (image.height+position[1]>overImage.height):
        # resize the overImage if necessary
        newImg=Image.new(
            size=(
                int(max(image.width+position[0],overImage.width)),
                int(max(image.height+position[1],overImage.height))),
            mode=imageRepr.maxMode(image,overImage,requireAlpha=True))
        # if we are negative position, we need to shift back by that amount
        positionShift=(abs(min(position[0],0)),abs(min(position[1],0)))
        position=(max(0,position[0]),max(0,position[1]))
        paste(overImage,newImg,positionShift)
        overImage=newImg
    elif hasattr(overImage,'immutable') and overImage.immutable:
        # if flagged immutable, create a copy that we are allowed to change
        overImage=overImage.copy()
    # do the deed
    position=(int(position[0]),int(position[1]))
    if imageRepr.hasAlpha(image):
        # TODO: which of these two lines is best?
        #overImage.paste(image,position,image) # NOTE: (image,(x,y),alphaMask)
        if overImage.mode!=image.mode:
            overImage=overImage.convert(image.mode)
        overImage.alpha_composite(image,dest=position)
    else:
        overImage.paste(image,position)
    return overImage


def getSize(ofThis):
    """
    get the xy size of something regardless of what kind it is

    :param ofThis: can be any of
        PIL image
        numpy array
        (w,h) or [w,h]
        (x,y,w,h) or [x,y,w,h]
        "w,h"
        anything derived from Bounds
    """
    if isinstance(ofThis,str):
        return ofThis.split(',')[0:2]
    if isinstance(ofThis,(tuple,list)):
        if len(ofThis)==2:
            return [ofThis[-2],ofThis[-1]]
    if isinstance(ofThis,np.ndarray):
        return [ofThis.shape[1],ofThis.shape[0]]
    if isinstance(ofThis,Bounds):
        return ofThis.size
    if isinstance(ofThis,Image.Image):
        return [ofThis.width,ofThis.height]
    return None


def makeSameSize(img1,img2,edgeFill='#ffffff00'):
    """
    pad the images with so they are the same size.  Commonly used for
    things like resizing before blending

    :param img1: first image
    :param img2: second image
    :param edgeFill: defines how to extend.  It can be:
        mirror - reflect the pixels leading up to the border
        repeat - repeat the image over again (useful with repeating textures)
        clamp - streak last pixels out to edge
        [background color] - simply fill with the given color

    :return: (img1,img2)
    """
    size1=getSize(img1)
    size2=getSize(img2)
    if size1[0]<size2[0] or size1[0]<size2[0]:
        img1=extendImageCanvas(img1,
            (max(size1[0],size2[0]),max(size1[1],size2[1])),edgeFill)
    if size2[0]<size1[0] or size2[0]<size1[0]:
        img2=extendImageCanvas(img2,
            (max(size1[0],size2[0]),max(size1[1],size2[1])),edgeFill)
    return img1,img2


def bestBounds(image,size,regionOfInterest=None):
    """
    utility determine the best bounds for a given size
    if regionOfInterest is None, then will center the crop

    :param image: any supported image type
    :param size: can be anything that getSize() supports
    :param regionOfInterest: a black and white mask used to
        liquid resize images
        (if =True, then auto-calculate as necessary)

    :return (x,y,x2,y2):
    """
    imgSize=getSize(image)
    size=getSize(size)
    dx=max(int(imgSize[0]-size[0]),0)
    dy=max(int(imgSize[1]-size[1]),0)
    if regionOfInterest is None:
        return (dx/2,dy/2,imgSize[0]-dx/2,imgSize[1]-dy/2)
    if regionOfInterest is True: # auto-calculate
        from . import autoInterest
        regionOfInterest=autoInterest.interest(image)
    # determine bounds
    n,s,e,w=0,imgSize[0]-1,imgSize[1]-1,0
    if dy>0:
        col_roi=np.sum(regionOfInterest,axis=1)
        ff=False # when there's a tie, alternate sides
        for _ in range(dy):
            if col_roi[e]>col_roi[w]:
                w+=1
                ff=True
            elif col_roi[e]<col_roi[w] or ff:
                e-=1
                ff=False
            else:
                w+=1
                ff=True
    if dx>0:
        col_roi=np.sum(regionOfInterest,axis=0)
        ff=False # when there's a tie, alternate sides
        for _ in range(dx):
            if col_roi[s]>col_roi[n]:
                n+=1
                ff=True
            elif col_roi[s]<col_roi[n] or ff:
                s-=1
                ff=False
            else:
                n+=1
                ff=True
    return (n,w,s,e)


def crop(image,size,regionOfInterest=None):
    """
    crop an image to a given size

    :param image: any supported image type
    :param size: can be anything that getSize() supports
    :param regionOfInterest: black and white mask used to liquid resize images
        (if =True, then auto-calculate as necessary)

    :returns: image of the cropped size (or smaller)
        can return None if selection is of zero size
    """
    image=imageRepr.numpyArray(image)
    if not isinstance(size,(list,tuple)) \
        and (not isinstance(size,np.ndarray) or len(size.shape)>1):
        #
        size=getSize(size)
    if isinstance(size,tuple): # make editable
        size=[wh for wh in size]
    imageSize=getSize(image)
    if imageSize==size: # no resizing necessary
        return image
    if len(size)==2:
        size=bestBounds(image,size,regionOfInterest)
    image=image[size[1]:size[3],size[0]:size[2]]
    return image


def stretch(image,toSize,interpolation=None,regionOfInterest=None):
    """
    stretch an image to the given size

    :param toSize: the actual size to stretch to.  If one value is None,
        then the aspect ratio is maintained
    :param interpolation: 'nearest','lanczos','bilinear','bicubic','cubic',
        'liquid' or a numeric order for spline interpolation
        if None, attempt to choose the best.  Generally you should keep it
        on that None or 'liquid'.
    :param regionOfInterest: black and white mask used to liquid resize images
        (if =True, then auto-calculate as necessary)

    NOTE: if toSize matches the aspect ratio of the image, will ALWAYS change
        to bilinear for speed.

    TODO: liquid rescale not yet implemented
    """
    image=imageRepr.numpyArray(image)
    if image is None:
        return image
    # figure out sizes of things
    size=image.shape[0:2]
    aspect=size[0]/size[1]
    if toSize[0] is None:
        if toSize[1] is None:
            return image
        toSize[0]=toSize[1]*aspect
        newAspect=aspect
    elif toSize[1] is None:
        toSize[1]=toSize[0]/aspect
        newAspect=aspect
    else:
        newAspect=toSize[0]/toSize[1]
    # figure out resizing method
    if aspect==newAspect:
        interpolation='bilinear'
    elif interpolation is None:
        if size[0]>toSize[0] or size[1]>toSize[1]:
            interpolation='lanczos'
        else:
            interpolation='bilinear'
    # perform the correct resize
    if isinstance(interpolation,str):
        if interpolation in ['nearest','lanczos','bilinear','bicubic']:
            resample:typing.Any=None
            if interpolation=='nearest':
                resample=Image.NEAREST # pylint: disable=no-member
            elif interpolation=='bilinear':
                resample=Image.BILINEAR # pylint: disable=no-member
            elif interpolation=='bicubic':
                resample=Image.BICUBIC # pylint: disable=no-member
            elif interpolation=='lanczos':
                resample=Image.LANCZOS # pylint: disable=no-member
            else:
                resample=Image.NEAREST # pylint: disable=no-member
            return imageRepr.numpyArray(
                imageRepr.pilImage(image).resize(
                    (int(toSize[0]),int(toSize[1])),
                    resample))
        if interpolation=='liquid':
            if regionOfInterest is True:
                from . import autoInterest
                regionOfInterest=autoInterest.interest(image)
            raise NotImplementedError()
    interpolation=int(interpolation)
    scale=(size[0]/toSize[0],size[1]/toSize[1])
    return scipy.ndimage.zoom(
        image,scale,order=interpolation,interpolation='reflect')


def scale(
    image,
    toSize=None,
    scale=None,
    regionOfInterest=None,
    aspect:str='stretch',
    interpolation=None,
    exactSize:bool=True,
    edgeFill="#ffffff00"):
    """
    :param image: image to scale.  Can be PIL image, numpy array, path, or
        supported url
    :param toSize: scale to fit a given size.  If missing, must have scale.
    :param scale: scale by a percent. If missing, must have toSize.
    :param aspect: can be:
        'stretch' - do not maintain aspect
        'minimize' - fit entirely within the new image (if exactSize then fill
            any gaps with edgeFill)
        'maximize' - grow to fill entire new image (if exactSize then crop to
            the requested size)
    :param interpolation: 'nearest','lanczos','bilinear','bicubic','cubic',
        'liquid' or a numeric order for spline interpolation
        if None, attempt to choose the best.  Generally you should keep it
        on that None or 'liquid'.
    :param exactSize: crop and/or fill to fit the exact size specified.
        If False, then aspect!='stretch' can create an image of a
        different size.
    :param regionOfInterest: a black and white mask used to smartly crop or
        liquid resize images If =True, then auto-generate.
    :param edgeFill: defines how to extend.  It can be:
        ignore - ignore edge. Make the image the closest size and ignore if it
            isn't the size requested
        mirror - reflect the pixels leading up to the border
        repeat - repeat the image over again (useful with repeating textures)
        clamp - streak last pixels out to edge
        [background color] - simply fill with the given color
        # TODO: not fully handled!
    """
    image=imageRepr.numpyArray(image)
    if image is None:
        return image
    size=getSize(image)
    # figure out the sizes
    if toSize is None:
        if scale is None:
            return image
        toSize=[scale[0]*size[0],scale[1]*size[1]]
        if isinstance(scale,tuple):
            scale=[scale[0],scale[1]]
    else:
        scale=[toSize[0]/size[0],toSize[1]/size[1]]
        if isinstance(toSize,tuple):
            toSize=[toSize[0],toSize[1]]
    if toSize==size:
        return image
    # adjust things if we are attempting to maintain aspect ratio
    if aspect!='stretch' and toSize[0] is not None and toSize[1] is not None:
        aspectRatio=float(size[0])/size[1]
        if aspect=='minimize':
            if size[0]/toSize[0]>size[1]/toSize[1]:
                tmpSize=(toSize[0],toSize[0]/aspectRatio)
            else:
                tmpSize=(toSize[1]*aspectRatio,toSize[1])
        elif aspect=='maximize':
            if size[0]/toSize[0]<size[1]/toSize[1]:
                tmpSize=(toSize[0],toSize[0]/aspectRatio)
            else:
                tmpSize=(toSize[1]*aspectRatio,toSize[1])
        else:
            raise IndexError('ERR: Unknown aspect setting')
        if regionOfInterest is True:
            from . import autoInterest
            regionOfInterest=autoInterest.interest(image)
        image=stretch(image,tmpSize,interpolation,regionOfInterest)
        if exactSize: # crop/fill as necessary
            if tmpSize[0]>toSize[0] or tmpSize[1]>toSize[1]:
                image=crop(image,toSize,regionOfInterest)
            if tmpSize[0]<toSize[0] or tmpSize[1]<toSize[1]:
                bx=toSize[0]-tmpSize[0]/2
                by=toSize[1]-tmpSize[1]/2
                image=imageBorder(image,(bx,by),edgeFill)
    else: # aspect='stretch' can perform a simple image stretch
        image=stretch(image,toSize,interpolation,regionOfInterest)
    return image


def cmdline(args):
    """
    Run the command line

    :param args: command line arguments (WITHOUT the filename)
    """
    from . import helper_routines
    printHelp=False
    if not args:
        printHelp=True
    else:
        edgeFill=(128,128,128,0)
        aspect='stretch'
        regionOfInterest=None
        exactSize=True
        interpolation=None
        for arg in args:
            if arg.startswith('-'):
                arg=[a.strip() for a in arg.split('=',1)]
                if arg[0] in ['-h','--help']:
                    printHelp=True
                elif arg[0]=='--edgeFill':
                    if len(arg)>1:
                        edgeFill=arg[1]
                elif arg[0]=='--aspect':
                    if len(arg)>1:
                        aspect=arg[1]
                elif arg[0]=='--interpolation':
                    if len(arg)>1:
                        if arg[1].lower() in ['none','no','n']:
                            interpolation=None
                        else:
                            interpolation=arg[1]
                elif arg[0]=='--exactSize':
                    if len(arg)>1:
                        exactSize=arg[1][0].lower() in ['t','1','y']
                elif arg[0]=='--regionOfInterest':
                    if len(arg)>1:
                        if arg[1].lower() in ['true','t','1','yes','y']:
                            regionOfInterest=True
                        elif arg[1].lower() in \
                            ['false','f','0','no','n','none']:
                            regionOfInterest=None
                        else:
                            regionOfInterest=arg[1]
                # commands
                elif arg[0]=='--imageBorder':
                    if len(arg)>1:
                        size=[float(x) for x in arg[1].split(',')]
                        image=imageBorder(image,size,edgeFill)
                elif arg[0]=='--scale':
                    if len(arg)>1:
                        toScale=None
                        toSize=[float(x) for x in arg[1].split(',')]
                        if (toSize[0] is not None and toSize[0]<1.0) \
                            or (toSize[1] is not None and toSize[1]<1.0):
                            #
                            toScale=toSize
                            toSize=None
                        image=scale(image,toSize,toScale,regionOfInterest,
                            aspect,interpolation,exactSize,edgeFill)
                elif arg[0]=='--crop':
                    if len(arg)>1:
                        size=[float(x) for x in arg[1].split(',')]
                        image=crop(image,size,regionOfInterest)
                elif arg[0]=='--stretch':
                    if len(arg)>1:
                        size=[float(x) for x in arg[1].split(',')]
                        image=stretch(
                            image,size,interpolation,regionOfInterest)
                elif arg[0]=='--bestBounds':
                    if len(arg)>1:
                        size=[float(x) for x in arg[1].split(',')]
                        print(bestBounds(image,size,regionOfInterest))
                elif arg[0]=='--show':
                    helper_routines.preview(image)
                elif arg[0]=='--save':
                    if len(arg)>1:
                        imageRepr.pilImg(image).save(arg[1])
                else:
                    print('ERR: unknown argument "'+arg[0]+'"')
            else:
                image=imageRepr.defaultLoader(arg)
    if printHelp:
        print('Usage:')
        print('  resizing.py image.jpg [options] [commands]')
        print('Options:')
        print('   --edgeFill= .............. how to extend. It can be:')
        print('\t  ignore  - ignore edge. Make the image the closest size and')
        print('\t            ignore if it isn\'t the size requested')
        print('\t  mirror  - reflect the pixels leading up to the border')
        print('\t  repeat  - repeat the image over again')
        print('\t            (useful with repeating textures)')
        print('\t  clamp   - streak last pixels out to edge')
        print('\t  [color] - simply fill with the given color')
        print('   --aspect= ................. what to do about image aspect')
        print('                               ratio. can be:')
        print('\t  stretch - do not maintain aspect')
        print('\t  minimize - fit entirely within the new image')
        print('\t             (if exactSize then fill any gaps with edgeFill)')
        print('\t  maximize - grow to fill entire new image (if exactSize')
        print('\t             then crop to the requested size)')
        print('   --interpolation= .......... how to resize pixels')
        print('\t  nearest,lanczos,bilinear,bicubic,cubic,liquid or a numeric')
        print('\t  order for spline interpolation')
        print('\t  If missing, attempt to choose the best.  Generally you')
        print('    should keep it on that or liquid')
        print('   --exactSize=[T/F] ......... crop and/or fill to fit the')
        print('    exact size specified.')
        print('\t  If False, then maintainAspect=True can create an image of')
        print('\t a different size.')
        print('   --regionOfInterest= ....... a black and white mask used to')
        print('   liquid resize images')
        print('\t  (or =True, then auto-calculate as necessary)')
        print('Commands:')
        print('   --imageBorder=thick .. expand the image with a border - ')
        print('     either a size, (horizontal,vertical), or (n,s,e,w)')
        print('   --scale=amtX,amtY .... if amt<1.0, then scale by percent,')
        print('     otherwise scale to exact size')
        print('   --crop=w,h ........... crop to w,h')
        print('   --stretch=w,h ........ stretch to w,h')
        print('   --bestBounds=w,h ..... figure out and print(the best bounds')
        print('     (for a region of interest)')
        print('   --show ............... show the working image')
        print('   --save=filename ...... save the working image')
        print('Notes:')
        print('   * All filenames can also take urls like')
        print('      file:// http:// https:// ftp://')


if __name__=='__main__':
    import sys
    cmdline(sys.argv[1:])

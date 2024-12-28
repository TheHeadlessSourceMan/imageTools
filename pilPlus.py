#!/usr/bin/env
# -*- coding: utf-8 -*-
"""
This is an experimental extension of PIL images allowing advanced access
"""
import typing
try:
    # first try to use bohrium, since it could help us accelerate
    # https://bohrium.readthedocs.io/users/python/
    import bohrium as np # type: ignore
except ImportError:
    # if not, plain old numpy is good enough
    import numpy as np
import PIL.Image
from .imageRepr import defaultLoader,numpyArray
from .bounds import Bounds
from .colorSpaces import grayscale
from .numberSpaces import numberspaceTransform


class PilPlusImage(PIL.Image.Image,Bounds):
    """
    This is an experimental extension of PIL images allowing advanced access
    """

    ANTIALIAS=PIL.Image.ANTIALIAS
    BILINEAR=PIL.Image.BILINEAR
    BICUBIC=PIL.Image.BICUBIC
    LINEAR=PIL.Image.LINEAR
    LANCZOS=PIL.Image.LANCZOS
    NEAREST=PIL.Image.NEAREST

    def __init__(self,image=None,imageLoader=None):
        self.imageLoader=imageLoader
        self._image=None
        PIL.Image.Image.__init__(self)
        Bounds.__init__(self)
        self._activeChannels={}
        if image is not None:
            self.image=image

    @property
    def w(self)->float:
        """
        changing width value changes the image size
        """
        img=self.image
        if img is not None:
            return img.width
        return 0
    @w.setter
    def w(self,w:float):
        if w is None or w==0:
            self.image=None
        else:
            img=self.image
            if img is not None and img.width!=w:
                self.image=img.resize((w,img.height))

    @property
    def h(self)->float:
        """
        changing height value changes the image size
        """
        img=self.image
        if img is not None:
            return img.height
        return 0
    @h.setter
    def h(self,h:float):
        if h is None or h==0:
            self.image=None
        else:
            img=self.image
            if img is not None and img.height!=h:
                self.image=img.resize((img.width,h))

    @property
    def size(self)->typing.Tuple[float,float]:
        """
        changing size value changes the image size
        """
        return self.image.size
    @size.setter
    def size(self,size:typing.Tuple[float,float]):
        if self.image is not None:
            self.image=self.image.resize(size)

    def imath(self,mathstring:str):
        """
        calculate a math operation string on an image

        :param mathstring: a string operation to whip on this image
        :return: a new image
        """
        ret=self
        predicate=None
        mathstring=mathstring.split('.') # TODO: could get confused with decimals.
        for txstep in mathstring:
            if predicate is not None:
                txstep=predicate+'.'+txstep
                ret=numberspaceTransform(
                    ret,txstep,invert=False,complex=False,level=None,mode=None)
            else:
                ret=self.getChannel(txstep)
        return ret

    def getChannel(self,channelId:str):
        """
        get a channel in floating point form
        """
        if channelId not in self._activeChannels:
            if channelId in ('r','g','b') \
                or (channelId=='a' and len(self._activeChannels)<2):
                #
                if 'v' in self._activeChannels:
                    return self._activeChannels['v']
                res=numpyArray(self.image,floatingPoint=True,loader=None)
                if isinstance(res.size,int) or len(res.size)<3: # L
                    self._activeChannels['v']=res
                    return self._activeChannels['v']
                if res.size[2]==2: # LA
                    self._activeChannels['v']=res[:,:,0]
                    self._activeChannels['a']=res[:,:,1]
                elif res.size[2]==3: # RGB
                    self._activeChannels['r']=res[:,:,0]
                    self._activeChannels['g']=res[:,:,1]
                    self._activeChannels['b']=res[:,:,2]
                elif res.size[2]==3: # RGBA
                    self._activeChannels['r']=res[:,:,0]
                    self._activeChannels['g']=res[:,:,1]
                    self._activeChannels['b']=res[:,:,2]
                    self._activeChannels['a']=res[:,:,3]
            elif channelId=='a':
                ret=np.array(self.image.size())
                ret.fill(1.0)
                self._activeChannels['a']=ret
                return ret
        return self._activeChannels[channelId]

    def grayscale(self,method:str='BT.709'):
        """
        Convert to grayscale

        :param method: available are:
            "max" - max of all channels
            "min" - min of all channels
            "avg" - average of all channels
            "HSV" or "median" - average of max and min
            "app" - weighted average used by most image applications
                (photoshop,gimp,...)
            "BT.709" - [default] weighted average specified by
                ITU-R "luma" specification BT.709
            "BT.601" - weighted average specified by
                ITU-R spec BT.601 used by video formats
            [%R,%G,%B] - array of weight values

        NOTE:
            There are many different ways to go about this.
            http://www.tannerhelland.com/3643/grayscale-image-algorithm-vb6/
        """
        return grayscale(self.image,method)

    @property
    def image(self):
        """
        the wrapped PIL image
        """
        return self._image
    @image.setter
    def image(self,image):
        self._activeChannels={}
        if image is None:
            pass
        elif isinstance(image,PilPlusImage):
            image=image.image
        elif not isinstance(image,PIL.Image.Image):
            if not hasattr(self,'imageLoader'):
                msg=str(self.__class__.__name__)+str(dir(self))
                raise Exception(msg)
            if self.imageLoader is not None:
                image=self.imageLoader(image)
            else:
                image=defaultLoader(image)
        self._image=image

    @property
    def im(self):
        """
        needed for PIL compatibility
        """
        return self.image.im
    @im.setter
    def im(self,im):
        if self._image is None:
            self._image=PIL.Image.Image()
        self.image.im=im

    @property
    def filename(self)->str:
        """
        the name of the file (setting this implies load)
        """
        if self.image is None:
            return None
        if hasattr(self.image,'filename'):
            return self.image.filename
        return None
    @filename.setter
    def filename(self,filename:str):
        self.load(filename)

    @property
    def mode(self)->str:
        """
        The PIL color mode of the image pixels
        """
        return self.image.mode
    @mode.setter
    def mode(self,mode:str):
        if hasattr(self,'_image'):
            self.image.mode=mode

    def decode(self,
        data:bytes,
        imageType:typing.Optional[str]=None,
        useBase64:bool=False):
        """
        decode the data bytes
        """
        import io
        data=data.decode('utf-8')
        if useBase64:
            import base64
            data=base64.b64decode(data)
        f=io.BytesIO(data)
        if imageType is not None:
            img=PIL.Image.open(f,formats=[imageType])
        else:
            img=PIL.Image.open(f)
        f.close()
        return img

    def encode(self,
        asImageType:typing.Optional[str]=None,
        useBase64:bool=False
        )->bytes:
        """
        encode into bytes
        """
        import io
        if asImageType is None:
            asImageType='image/png'
        f=io.BytesIO()
        self.image.save(f,format=asImageType.rsplit('/',1)[-1])
        data=f.getvalue()
        f.close()
        if useBase64:
            import base64
            data=base64.b64encode(data)
        return data

    def toDataUrl(self,mimeType:str='image/png'):
        """
        this image as a data: url for embedding in html

        see also:
            https://en.wikipedia.org/wiki/Data_URI_scheme
        """
        data=self.encode(mimeType,useBase64=True)
        return "data:%s;base64,%s"%(mimeType,data.decode('ascii'))

    def assign(self,sizeOrImage)->None:
        """
        assign this image to another image, or a certain size
        """
        if isinstance(sizeOrImage,(list,tuple)) and len(sizeOrImage) in (2,4):
            Bounds.assign(self,sizeOrImage)
        elif isinstance(sizeOrImage,PilPlusImage):
            self.image=sizeOrImage.image
        elif isinstance(sizeOrImage,PIL.Image.Image):
            self.image=sizeOrImage
        else:
            Bounds.assign(self,sizeOrImage)

    def __repr__(self)->str:
        """
        string representation of this object
        """
        return '\n\t'.join((
            f'Image {Bounds.__repr__(self)}',
            f'mode: {self.mode}',
            f'source: {self.filename}'
            ))

    @property
    def R(self):
        """
        red channel
        """
        return self.getChannel('r')
    @property
    def G(self):
        """
        green channel
        """
        return self.getChannel('g')
    @property
    def B(self):
        """
        blue channel
        """
        return self.getChannel('b')
    @property
    def A(self):
        """
        alpha channel
        """
        return self.getChannel('a')
    @property
    def mask(self):
        """
        alpha channel
        """
        return self.getChannel('a')

    @property
    def numpyArray(self):
        """
        quick shortcut to get numpy array
        """
        return numpyArray(self.image)
    @numpyArray.setter
    def numpyArray(self,numpyArray):
        self.image=numpyArray

    # ===== PIL image routines
    def fromstring(self,*args,**kw):
        """ delegates to the wrapped image """
        self._image.fromstring(*args,**kw)
    def tostring(self,*args,**kw):
        """ delegates to the wrapped image """
        return self._image.tostring(*args,**kw)
    def offset(self,xoffset,yoffset=None):
        """ delegates to the wrapped image """
        self._image.offset(xoffset,yoffset)
    def alpha_composite(self,*args,**vargs):
        return self.image.alpha_composite(*args,**vargs)
    def category(self,*args):
        return self.image.category(*args)
    def close(self,*args):
        return self.image.close(*args)
    def convert(self,*args,**vargs):
        return self.image.convert(*args,**vargs)
    def copy(self,*args):
        return self.image.copy(*args)
    def crop(self,*args):
        return self.image.crop(*args)
    def draft(self,*args):
        return self.image.draft(*args)
    def effect_spread(self,*args):
        return self.image.effect_spread(*args)
    def filter(self,*args):
        return self.image.filter(*args)
    def format(self,*args):
        return self.image.format(*args)
    def format_description(self,*args):
        return self.image.format_description(*args)
    def frombytes(self,*args):
        return self.image.frombytes(*args)
    def fromstring(self,*args):
        return self.image.fromstring(*args)
    def getbands(self,*args):
        return self.image.getbands(*args)
    def getbbox(self,*args):
        return self.image.getbbox(*args)
    def getchannel(self,*args):
        return self.image.getchannel(*args)
    def getcolors(self,*args):
        return self.image.getcolors(*args)
    def getdata(self,*args):
        return self.image.getdata(*args)
    def getextrema(self,*args):
        return self.image.getextrema(*args)
    def getim(self,*args):
        return self.image.getim(*args)
    def getpalette(self,*args):
        return self.image.getpalette(*args)
    def getpixel(self,*args):
        return self.image.getpixel(*args)
    def getprojection(self,*args):
        return self.image.getprojection(*args)
    #def height(self,*args): # get from bounds
    #\treturn self.image.height(*args)
    def histogram(self,*args):
        return self.image.histogram(*args)
    #def im(self,*args):
    #\treturn self.image.im(*args)
    def info(self,*args):
        return self.image.info(*args)
    def load(self,*args,**vargs):
        """
        load an image file
        """
        return self.image.load(*args,**vargs)
    #def mode(self,*args):
    #\treturn self.image.mode(*args)
    def palette(self,*args):
        return self.image.palette(*args)
    def paste(self,*args):
        return self.image.paste(*args)
    def point(self,*args):
        return self.image.point(*args)
    def putalpha(self,*args):
        return self.image.putalpha(*args)
    def putdata(self,*args):
        return self.image.putdata(*args)
    def putpalette(self,*args):
        return self.image.putpalette(*args)
    def putpixel(self,*args):
        return self.image.putpixel(*args)
    def pyaccess(self,*args):
        return self.image.pyaccess(*args)
    def quantize(self,*args):
        return self.image.quantize(*args)
    def readonly(self,*args):
        return self.image.readonly(*args)
    def remap_palette(self,*args):
        return self.image.remap_palette(*args)
    def resize(self,*args):
        return self.image.resize(*args)
    def rotate(self,*args):
        return self.image.rotate(*args)
    def save(self,*args):
        return self.image.save(*args)
    def seek(self,*args):
        return self.image.seek(*args)
    def show(self,*args):
        return self.image.show(*args)
    #def size(self,*args): # get from bounds
    #\treturn self.image.size(*args)
    def split(self,*args):
        return self.image.split(*args)
    def tell(self,*args):
        return self.image.tell(*args)
    def thumbnail(self,*args):
        return self.image.thumbnail(*args)
    def tobitmap(self,*args):
        return self.image.tobitmap(*args)
    def tobytes(self,*args):
        return self.image.tobytes(*args)
    def toqimage(self,*args):
        return self.image.toqimage(*args)
    def toqpixmap(self,*args):
        return self.image.toqpixmap(*args)
    def transform(self,*args):
        return self.image.transform(*args)
    def transpose(self,*args):
        return self.image.transpose(*args)
    def verify(self,*args):
        return self.image.verify(*args)
    #def width(self,*args): # get from bounds
    #\treturn self.image.width(*args)


def cmdline(args:typing.List[str])->int:
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
        print('  pilPlus.py [options]')
        print('Options:')
        print('   NONE')
        return -1
    return 0


if __name__=='__main__':
    import sys
    cmdline(sys.argv[1:])

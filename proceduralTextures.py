#!/usr/bin/env
# -*- coding: utf-8 -*-
"""
Routines for the creation of procedural textures

TODO: see also:
    https://developer.blender.org/diffusion/B/browse/master/source/blender/render/intern/source/render_texture.c

"""
import math
try:
    # first try to use bohrium, since it could help us accelerate
    # https://bohrium.readthedocs.io/users/python/
    import bohrium as np
except ImportError:
    # if not, plain old numpy is good enough
    import numpy as np
import scipy.ndimage
import scipy.signal
from PIL import Image
from .helper_routines import *
from .selectionsAndPaths import *


def perlinNoiseX(size=(256,256),seed=None):
    """
    Generate a black and white image consisting of perlinNoise (clouds)
    """
    def lerp(a,b,x):
        "linear interpolation"
        return a + x * (b-a)
    def fade(t):
        "6t^5 - 15t^4 + 10t^3"
        return 6 * t**5 - 15 * t**4 + 10 * t**3
    def gradient(h,x,y):
        "converts h to the right gradient vector and return the dot product with (x,y)"
        vectors = np.array([[0,1],[0,-1],[1,0],[-1,0]])
        g = vectors[h%4]
        return g[:,:,0] * x + g[:,:,1] * y
    # permutation table
    if seed is not None:
        np.random.seed(seed)
    p = np.arange(256,dtype=int)
    np.random.shuffle(p)
    p = np.stack([p,p]).flatten()
    # coordinates of the top-left
    xi = size[0].astype(int)
    yi = size[1].astype(int)
    # internal coordinates
    xf = size[0] - xi
    yf = size[1] - yi
    # fade factors
    u = fade(xf)
    v = fade(yf)
    # noise components
    n00 = gradient(p[p[xi]+yi],xf,yf)
    n01 = gradient(p[p[xi]+yi+1],xf,yf-1)
    n11 = gradient(p[p[xi+1]+yi+1],xf-1,yf-1)
    n10 = gradient(p[p[xi+1]+yi],xf-1,yf)
    # combine noises
    x1 = lerp(n00,n10,u)
    x2 = lerp(n01,n11,u)
    return lerp(x1,x2,v)
def callPerlinX(seed=None):
    """
    utility to generate perlin noise
    """
    lin = np.linspace(0,5,256,endpoint=False)
    x,y = np.meshgrid(lin,lin)
    img=perlinNoise(x,y,seed=seed)


def randomNoise(size=(256,256),seed=None):
    """
    generate a patch of random pixels
    """
    if seed is not None:
        np.random.seed(seed)
    return np.random.random(size)


def perlinNoiseY(imgx=256,imgy=256):
    """
    generate perlin noise in the y direction
    """
    import random
    image = Image.new("RGB", (imgx, imgy))
    pixels = image.load()
    octaves = int(math.log(max(imgx, imgy), 2.0))
    persistence = random.random()
    imgAr = [[0.0 for i in range(imgx)] for j in range(imgy)] # image array
    totAmp = 0.0
    for k in range(octaves):
        freq = 2 ** k
        amp = persistence ** k
        totAmp += amp
        # create an image from n by m grid of random numbers (w/ amplitude)
        # using Bilinear Interpolation
        n = freq + 1; m = freq + 1 # grid size
        ar = [[random.random() * amp for i in range(n)] for j in range(m)]
        nx = imgx / (n - 1.0); ny = imgy / (m - 1.0)
        for ky in range(imgy):
            for kx in range(imgx):
                i = int(kx / nx); j = int(ky / ny)
                dx0 = kx - i * nx; dx1 = nx - dx0
                dy0 = ky - j * ny; dy1 = ny - dy0
                z = ar[j][i] * dx1 * dy1
                z += ar[j][i + 1] * dx0 * dy1
                z += ar[j + 1][i] * dx1 * dy0
                z += ar[j + 1][i + 1] * dx0 * dy0
                z /= nx * ny
                imgAr[ky][kx] += z # add image layers together

    # paint image
    for ky in range(imgy):
        for kx in range(imgx):
            c = int(imgAr[ky][kx] / totAmp * 255)
            pixels[kx, ky] = (c, c, c)
    return image

def perlinNoise(size=(256,256),octaves=None,seed=None):
    """
    generate perlin noise
    """
    if octaves is None:
        octaves=math.log(max(size),2.0)
    ret=np.zeros(size)
    for n in range(1,int(octaves)):
        k=n
        amp=1.0#/octaves/float(n)
        px=randomNoise((size[0]/k,size[1]/k),seed)
        px=scipy.ndimage.zoom(px,(float(ret.shape[0])/px.shape[0],float(ret.shape[1])/px.shape[1]),order=2)
        px=np.clip(px,0.0,1.0) # unfortunately zoom function can cause values to go out of bounds :(
        ret=(ret+px*amp)/2.0 # average with existing
    ret=np.clip(ret,0.0,1.0)
    return ret

# TODO: don't know where to put this, but it looks important
#def f(points):
#\t\treturn np.sin(ret.index/w)
#ret=scipy.ndimage.generic_filter(ret,f,size=None,footprint=(1),output=None,mode='reflect',cval=0.0, origin=0, extra_arguments=(), extra_keywords=None)


def waveImage(size=(256,256),repeats=2,angle=0,wave='sine',radial=False):
    """
    create an image based on a sine, saw, or triangle wave function
    """
    ret=np.zeros(size)
    if radial:
        raise NotImplementedError() # TODO: Implement radial wave images
    else:
        twopi=2*pi
        thetas=np.arange(size[0])*float(repeats)/size[0]*twopi
        if wave=='sine':
            ret[:,:]=0.5+0.5*np.sin(thetas)
        elif wave=='saw':
            n=np.round(thetas/twopi)
            thetas-=n*twopi
            ret[:,:]=np.where(thetas<0,thetas+twopi,thetas)/twopi
        elif wave=='triangle':
            ret[:,:]=1.0-2.0*np.abs(np.floor((thetas*(1.0/twopi))+0.5)-(thetas*(1.0/twopi)))
    return ret


def randCurve(seed=None):
    """
    get a random number in a bell curve centered about 0
    """
    import random
    return random.random(seed)-random.random(seed)


def randomScatter(img,distance,seed=None):
    """
    randomly scatter pixels around
    """
    def scatPoint(point):
        return (point[0]+distance*randCurve(seed),point[1]+distance*randCurve(seed))
    img=scipy.ndimage.geometric_transform(img,scatPoint,mode='nearest')
    return img


def combine(img1,img2):
    """
    average to images together
    """
    if img1 is None:
        return img2
    if img2 is None:
        return img1
    return (img1+img2)/2.0


def section(img,minVal=0.0,maxVal=1.0):
    """
    clamp the values of an image to a given range
    """
    return normalize(np.clip(img,minVal,maxVal))


def filmGrain(img,amount=None,iso=None,grainSize=None,seed=None):
    """
    simulate film grain

    :param iso: try and mimic this film speed
    :param seed: random seed for repeatability

    NOTE:
        A better/more accurate way to do this is with an image
        of actual film grain called an FGO, or Film Grain Overlay.

    See also:
        https://hal.archives-ouvertes.fr/hal-01494123v2/document
        http://www.dctsystems.co.uk/Text/grain.pdf
    """
    if isinstance(img,tuple):
        size=img
        img=None
    else:
        size=img.size
    if grainSize is None:
        grainSize=(1200-iso)/300 # TODO: need to figure something out
    # TODO: this is not great, as it is uniform squares.  Something like vironoi
    #   would be better
    fgo=blocks(size,grainSize,seed)
    if img is None:
        img=fgo
    else:
        pass # TODO: do a grain overlay
    return img


def blocks(size=(256,256),blockSize=64,seed=None):
    """
    create randomly-shaded blocks of a given block size
    """
    return smoothNoise(size,undersize=blockSize/size[0],scaleMode=Image.NEAREST,seed=seed)


def deltaC(size,center=None,magnitude=False):
    """
    returns an array where every value is the distance from
    the center point.

    :param center: center point, if not specified, assume size/2
    :param magnitude: return a single magnitude value instead of [dx,dy]
    """
    size=(int(size[0]),int(size[1]))
    if center is None:
        center=size[0]/2.0,size[1]/2.0
    if magnitude:
        if False:
            # old way, non-vector
            img=np.ndarray((size[0],size[1]))
            for x in range(size[0]):
                for y in range(size[1]):
                    img[x,y]=math.sqrt((x-center[0])**2+(y-center[1])**2)
        else:
            img=np.sqrt((np.arange(size[0])-center[0])**2+(np.arange(size[1])[:,None]-center[1])**2)
    else:
        img=np.ndarray((size[0],size[1],2))
        for x in range(size[0]):
            for y in range(size[1]):
                img[x,y]=(x-center[0]),(y-center[1])
    return img


def clock(size=(256,256),angle=0,sharpEdge=False):
    """
    :param sharpEdge: if true, 100% black ends at 100% white for a difinitive angle -- otherwise gives more of a lighting effect
    """
    angle=0.75-angle/360.0
    dc=deltaC(size)
    img=np.arctan2(dc[:,:,0],dc[:,:,1])/math.pi
    img=(normalize(img)+angle)%1.0
    if not sharpEdge:
        img=deltaFromGray(img)
    return img


def distance(src):
    """
    calculate the distance transform upon an image
    """
    return scipy.ndimage.morphology.distance_transform_edt(src)


def xypoints(img):
    """
    get numpy array of x,y points for the image

    yeah, it may be a bad way to do this, yet here we are, Robert.
    """
    xy=np.ndarray((img.shape[0],img.shape[1],2))
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            xy[x,y]=[x,y]
    return xy


def voronoi(size=(256,256),npoints=30,mode='squared',invert=False,seed=None):
    """

    :param mode: 'simple','squared','twoNearestDiff','twoNearestMult'

    see also:
        http://blackpawn.com/texts/cellular/default.html

    TODO: this could be optomized usinf a cKDtree
        https://stackoverflow.com/questions/10818546/finding-index-of-nearest-point-in-numpy-arrays-of-x-and-y-coordinates
    """
    size=(int(size[0]),int(size[1]))
    img=np.ndarray(size)
    size=np.array(size)
    if seed is not None:
        np.random.seed(seed)
    points=np.random.rand(int(npoints),2)*size # random points where voroni connections will appear
    xy=xypoints(img).reshape((-1,2)) # flattened list of xy points for every pixel
    dist=scipy.spatial.distance.cdist(xy,points) # for each pixel, determine distance to all voroni points
    if mode=='simple':
        img=np.min(dist,axis=1)
    elif mode=='squared':
        img=np.min(dist,axis=1)**2
    elif mode=='twoNearestDiff':
        #smallest=np.argsort(dist,axis=1)
        smallest=np.argpartition(dist,2,axis=1)[:,0:2] # faster because it stops sorting after 2
        img=dist[:,smallest[:,1]]-dist[:,smallest[:,0]]
    elif mode=='twoNearestMult':
        #smallest=np.argsort(dist,axis=1)
        smallest=np.argpartition(dist,2,axis=1)[:,0:2] # faster because it stops sorting after 2
        img=dist[:,smallest[:,1]]*dist[:,smallest[:,0]]
    else:
        raise NotImplementedError('mode="'+mode+'"')
    # convert pixel list back into 2d array
    img=normalize(img.reshape((size[0],size[1])))
    if invert:
        img=1.0-img
    return img


def smoothNoise(size=(256,256),undersize=0.5,scaleMode=None,seed=None):
    """
    create smooth noise by starting with undersized random noise
    and then scaling it up

    :param size: finished size
    :param undersize: start undersized (percent) then scale up
    :param scaleMode: PIL scale mode (default is BILINEAR)
    :param seed: random seed to permit repeatability
    """
    if seed is not None:
        np.random.seed(seed)
    if scaleMode is None:
        scaleMode=Image.BILINEAR
    noise=np.random.random((int(size[0]*undersize),int(size[0]*undersize)))
    if undersize!=0:
        size=(int(size[0]),int(size[1]))
        noise=np.array(Image.fromarray(noise).resize(size,scaleMode))/255.0
    return noise


def turbulence(size=(256,256),turbSize=None,seed=None):
    """
    Generate a turbulence noise

    :param turbSize: initial size of the turbulence default is max(w,h)/10
    """
    if turbSize is None:
        turbSize=max(size)/10.0
    img=None
    s=turbSize
    while s>=1.0:
        noise=smoothNoise(size,s/turbSize,seed=seed)/s
        if img is None:
            img=noise
        else:
            img=img+noise
        s/=2.0
    return normalize(img)


def marble(size=(256,256),xPeriod=3.0,yPeriod=10.0,turbPower=0.8,turbSize=None,seed=None):
    """
    Generate a marble texture akin to blender and others

    :param xPeriod: defines repetition of marble lines in x direction
    :param yPeriod: defines repetition of marble lines in y direction
    :param turbPower: add twists and swirls (when turbPower==0  it becomes a normal sine pattern)
    :param turbSize: initial size of the turbulence default is max(w,h)/10

    NOTE:
        xPeriod and yPeriod together define the angle of the lines
        xPeriod and yPeriod both 0 then it becomes a normal clouds or turbulence pattern

    Optimized version of:
        https://lodev.org/cgtutor/randomnoise.html
    """
    w,h=size
    turb=turbulence(size,turbSize,seed=seed)*turbPower
    # determine where the pixels will cycle
    cycleValue=np.ndarray((w,h))
    for y in range(h):
        for x in range(w):
            cycleValue[x,y]=(x*xPeriod/w)+(y*yPeriod/h)
    # add in the turbulence and then last of all, make it a sinewave
    img=np.abs(np.sin((cycleValue+turb)*math.pi))
    return img


def woodRings(size=(256,256),xyPeriod=12.0,turbPower=0.15,turbSize=32.0,seed=None):
    """
    Draw wood rings

    :param xyPeriod: number of rings
    :param turbPower: makes twists
    :param turbSize: # initial size of the turbulence

    Optimized version of:
        https://lodev.org/cgtutor/randomnoise.html
    """
    turb=turbulence(size,turbSize,seed=seed)*turbPower
    dc=normalize(deltaC(size,magnitude=True))
    img = np.abs(np.sin(xyPeriod * (dc+turb) * math.pi))
    return normalize(img)


def waveformTexture(size=(256,256),waveform="sine",frequency=(10,10),noise=0.2,
    noiseBasis="perlin",noiseOctaves=4,noiseSoften=0.0,direction=0,invert=False,seed=None):
    """
    generate a texture by sweeping a waveform across the image
    """
    size=(int(size[0]),int(size[1]))
    w,h=size
    noiseSize=max(size)/8 # TODO: pass this value in somehow
    turb=turbulence(size,noiseSize,seed=seed)*noise
    if waveform.startswith('sine'):
        def sin(x):
            return np.abs(np.sin(x*math.pi))
        fn=sin
    elif waveform.startswith('tri'):
        fn=deltaFromGray
    elif waveform.startswith('saw'):
        def saw(x):
            return x
        fn=saw
    else:
        raise NotImplementedError()
    if direction=='circular':
        dc=normalize(deltaC(size,magnitude=True))
        xyPeriod=frequency[0]#(w/frequency[0]+h/frequency[1])/2.0
        img=fn(xyPeriod*(dc+turb))
        invert=invert is False # flip invert to look more as expected
    else:
        xPeriod=frequency[0]/2.0
        yPeriod=frequency[1]/2.0
        # determine where the pixels will cycle
        cycleValue=np.ndarray((w,h))
        for y in range(h):
            for x in range(w):
                cycleValue[x,y]=(x*xPeriod/w)+(y*yPeriod/h)
        # add in the turbulence and then last of all, make it a wave
        img=fn(cycleValue+turb)
    img=normalize(img)
    if invert:
        img=1.0-img
    return img

def clock2(size=(256,256),waveform="sine",frequency=(10,10),noise=0.2,noiseBasis="perlin",
    noiseOctaves=4,noiseSoften=0.0,direction=0,invert=False,seed=None):
    """
    a rotary clockface gradient
    """
    noiseSize=max(size)/8 # TODO: pass this value in somehow
    turb=turbulence(size,noiseSize,seed=seed)*noise
    #points=np.ndarray((int(size[0]),int(size[1])))
    angle=0.75-direction/360.0
    dc=deltaC(size)
    img=np.arctan2(dc[:,:,0],dc[:,:,1])/math.pi
    img=np.abs(np.sin((((normalize(img)+angle)%1.0)+turb)*math.pi))
    img=normalize(img)
    if invert:
        img=1.0-img
    return img

def x(img=(256,256),color=1):
    """
    img can be an image array or size tuple
    """
    if isinstance(img,tuple):
        size=img
        if isinstance(color,(float,int)):
            img=np.ndarray((size[0],size[1]))
        else:
            img=np.ndarray((size[0],size[1],len(color)))
    else:
        size=img.shape[0:1]
    img=line(img,[0,0],[size[0]-1,size[1]-1])
    img=line(img,[0,size[1]-1],[size[0]-1,0])
    return img

def line(img, xxx_todo_changeme, xxx_todo_changeme1,thick=1,color=1):
    """
    comes from this neat implementation:
        https://stackoverflow.com/questions/31638651/how-can-i-draw-lines-into-numpy-arrays
    """
    (x,y) = xxx_todo_changeme
    (x2,y2) = xxx_todo_changeme1
    def trapez(y,y0,w):
        return np.clip(np.minimum(y+1+w/2-y0, -y+1+w/2+y0),0,1)

    def weighted_line(r0, c0, r1, c1, w, rmin=0, rmax=np.inf):
        # The algorithm below works fine if c1 >= c0 and c1-c0 >= abs(r1-r0).
        # If either of these cases are violated, do some switches.
        if abs(c1-c0) < abs(r1-r0):
            # Switch x and y, and switch again when returning.
            xx, yy, val = weighted_line(c0, r0, c1, r1, w, rmin=rmin, rmax=rmax)
            return (yy, xx, val)

        # At this point we know that the distance in columns (x) is greater
        # than that in rows (y). Possibly one more switch if c0 > c1.
        if c0 > c1:
            return weighted_line(r1, c1, r0, c0, w, rmin=rmin, rmax=rmax)

        # The following is now always < 1 in abs
        num=r1-r0
        denom=c1-c0
        slope = np.divide(num,denom,out=np.zeros_like(denom), where=denom!=0)

        # Adjust weight by the slope
        w *= np.sqrt(1+np.abs(slope)) / 2

        # We write y as a function of x, because the slope is always <= 1
        # (in absolute value)
        x = np.arange(c0, c1+1, dtype=float)
        y = x * slope + (c1*r0-c0*r1) / (c1-c0)

        # Now instead of 2 values for y, we have 2*np.ceil(w/2).
        # All values are 1 except the upmost and bottommost.
        thickness = np.ceil(w/2)
        yy = (np.floor(y).reshape(-1,1) + np.arange(-thickness-1,thickness+2).reshape(1,-1))
        xx = np.repeat(x, yy.shape[1])
        vals = trapez(yy, y.reshape(-1,1), w).flatten()

        yy = yy.flatten()

        # Exclude useless parts and those outside of the interval
        # to avoid parts outside of the picture
        mask = np.logical_and.reduce((yy >= rmin, yy < rmax, vals > 0))

        return (yy[mask].astype(int), xx[mask].astype(int), vals[mask])
    def naive_line(r0, c0, r1, c1):
        # The algorithm below works fine if c1 >= c0 and c1-c0 >= abs(r1-r0).
        # If either of these cases are violated, do some switches.
        if abs(c1-c0) < abs(r1-r0):
            # Switch x and y, and switch again when returning.
            xx, yy, val = naive_line(c0, r0, c1, r1)
            return (yy, xx, val)

        # At this point we know that the distance in columns (x) is greater
        # than that in rows (y). Possibly one more switch if c0 > c1.
        if c0 > c1:
            return naive_line(r1, c1, r0, c0)

        # We write y as a function of x, because the slope is always <= 1
        # (in absolute value)
        x = np.arange(c0, c1+1, dtype=float)
        y = x * (r1-r0) / (c1-c0) + (c1*r0-c0*r1) / (c1-c0)

        valbot = np.floor(y)-y+1
        valtop = y-np.floor(y)
        return (np.concatenate((np.floor(y), np.floor(y)+1)).astype(int), np.concatenate((x,x)).astype(int),np.concatenate((valbot, valtop)))
    if thick==1:
        rows, cols, weights=naive_line(x,y,x2,y2)
    else:
        rows, cols, weights=weighted_line(x,y,x2,y2,thick,rmin=0,rmax=max(img.shape[0],img.shape[1]))
    w = weights.reshape([-1, 1]) # reshape anti-alias weights
    if len(img.shape)>2:
        img[rows, cols, 0:3] = (
            np.multiply((1 - w) * np.ones([1, 3]),img[rows, cols, 0:3]) +
            w * np.array([color])
        )
    else:
        img[rows, cols] = (
            np.multiply(
                (1 - w) * np.ones([1, ]),
                img[rows, cols]) +
            w *
            color
        )[0]

    return img


def grid(img=(256,256),nx=10,ny=None,thick=1,color='#000000'):
    """
    create a regular grid of points

    :param img: an image to draw a grid upon, or a size of image to create
    :param nx: Number of cells in the x direction
    :param ny: Number of cells in the y direction (if None, use same as nx)
    :param thick: grid line thickness in pixels
    :param color: grid line color
    """
    if isinstance(nx,str):
        nx=int(nx)
    if isinstance(ny,str):
        ny=int(ny)
    if isinstance(img,tuple):
        size=img
        if isinstance(color,(float,int)):
            img=np.ndarray((size[0],size[1]))
        else:
            img=np.ndarray((size[0],size[1],len(color)))
    else:
        size=img.shape[0:1]
    w=size[0]-1
    h=size[1]-1
    for x in range(0,w,w/nx):
        img=line(img,(x,0),(x,h),thick,color)
    for y in range(0,h,h/ny):
        img=line(img,(0,y),(w,y),thick,color)
    return img


def arbitraryWave(wave,fromT,toT,mirror=False):
    """
    evaluate arbitrary waveform between two points

    see also:
        https://docs.scipy.org/doc/scipy/reference/interpolate.html
    """
    pass


def cmdline(args):
    """
    Run the command line

    :param args: command line arguments (WITHOUT the filename)
    """
    printhelp=False
    if not args:
        printhelp=True
    else:
        img=None
        size=(256,256)
        seed=None
        mode='average'
        for arg in args:
            img2=None
            if arg.startswith('-'):
                arg=[a.strip() for a in arg.split('=',1)]
                if arg[0] in ['-h','--help']:
                    printhelp=True
                elif arg[0]=='--marble':
                    params=[size]
                    if len(arg)>1:
                        params.extend(arg[1].split(','))
                    img2=marble(*params,seed=seed)
                elif arg[0]=='--clouds':
                    params=[size]
                    if len(arg)>1:
                        params.extend(arg[1].split(','))
                    img2=perlinNoise(*params,seed=seed)
                elif arg[0]=='--voronoi':
                    params=[size]
                    if len(arg)>1:
                        params.extend(arg[1].split(','))
                    img2=voronoi(*params,seed=seed)
                elif arg[0]=='--woodRings':
                    params=[size]
                    if len(arg)>1:
                        params.extend(arg[1].split(','))
                    img2=woodRings(*params,seed=seed)
                elif arg[0]=='--blocks':
                    params=[size]
                    if len(arg)>1:
                        params.extend(arg[1].split(','))
                    img2=blocks(*params,seed=seed)
                elif arg[0]=='--x':
                    params=[size]
                    if len(arg)>1:
                        params.extend(arg[1].split(','))
                    img2=x(*params)
                elif arg[0]=='--grid':
                    params=[size]
                    if len(arg)>1:
                        params.extend(arg[1].split(','))
                    img2=grid(*params)
                elif arg[0]=='--wave':
                    params=[size]
                    if len(arg)>1:
                        params.extend(arg[1].split(','))
                    img2=waveformTexture(*params,seed=seed)
                elif arg[0]=='--clock':
                    params=[size]
                    if len(arg)>1:
                        params.extend(arg[1].split(','))
                    img2=clock2(*params)
                elif arg[0]=='--turbulence':
                    params=[size]
                    if len(arg)>1:
                        params.extend(arg[1].split(','))
                    img2=turbulence(*params,seed=seed)
                elif arg[0]=='--smooth':
                    params=[size]
                    if len(arg)>1:
                        params.extend(arg[1].split(','))
                    img2=smoothNoise(*params,seed=seed)
                elif arg[0]=='--random':
                    params=[size]
                    if len(arg)>1:
                        params.extend(arg[1].split(','))
                    img2=randomNoise(params,seed=seed)
                elif arg[0]=='--cartesian':
                    img=polar2cartesian(img)
                elif arg[0]=='--polar':
                    img=cartesian2polar(img)
                elif arg[0]=='--fft':
                    img=toFrequency(img)
                    size=(img.shape[0],img.shape[1])
                elif arg[0]=='--ifft':
                    img=fromFrequency(img)
                    size=(img.shape[0],img.shape[1])
                elif arg[0]=='--mode':
                    mode=arg[1]
                elif arg[0]=='--show':
                    preview(img)
                elif arg[0]=='--save':
                    pilImage(clampImage(img)).save(arg[1])
                else:
                    print('ERR: unknown argument "'+arg[0]+'"')
            else:
                img2=arg
            if img is None:
                img=img2
            elif img2 is None:
                if mode=='average':
                    img=combine(img,img2)
                elif mode=='displace':
                    img=displaceImage(img,img2)
            if not isinstance(img,np.ndarray):
                print(img.whatever)
    if printhelp:
        print('Usage:')
        print('  proceduralTextures.py [option|image.jpg]')
        print('Options:')
        print('   --show ................. show the image')
        print('   --marble[=xPeriod,yPeriod,turbPower,turbSize] .. marble')
        print('   --perlin ............... clouds')
        print('   --voronoi[=npoints,mode,invert] .. cell texture')
        print('   --woodRings[=xyPeriod,turbPower,turbSize] .. wood rings')
        print('...--blocks[=size] ........ randomly-shaded blocks ')
        print('   --x .................... big x across the image for testing')
        print('   --grid[=nx,ny] ......... grid')
        print('   --wave[=waveform,frequency,noise,noiseBasis,noiseOctaves,noiseSoften,direction,invert] .. a wave')
        print('   --clock[=waveform,frequency,noise,noiseBasis,noiseOctaves,noiseSoften,direction,invert] .. a smooth clockface or cone')
        print('   --turbulence[=turbSize] .. sea turbulence')
        print('   --smooth=[undersize] ..... smooth noise')
        print('   --polar ................ convert to polar space')
        print('   --cartesian ............ convert to cartesian space')
        print('   --fft .................. convert to frequency space')
        print('   --ifft ................. convert from frequency space')
        print('   --random ............... random static')
        print('   --mode=[average|displace] .. how to combine textures')
        print('...--seed=num ............. seed the randomizer for repeatable results')
        print('   --save=[filename] ...... save out the result')


if __name__=='__main__':
    import sys
    cmdline(sys.argv[1:])


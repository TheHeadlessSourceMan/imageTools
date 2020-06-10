#!/usr/bin/env
# -*- coding: utf-8 -*-
"""
This library manages holes in images

TOTALLY EXPERIMENTAL!
"""

class Hole(Selection):
    """
    Though a regular selection can contain holes, a hole itself
    has no internal holes. TODO: or does it!??
    """

    def __init__(self):
        self.boundingBox=None
        self.boundingRadius=None
        self.border=None
        
    @property
    def borderLoop(self):
        """
        note that loop is in order and contiguious
        """
        return self.points
        
    def normals(self,inside=False):
        """
        return the outside or inside normals for each point in the border
        
        returns [angle,point]
        """
        if invert:
            return self.pointAngles(90)
        return self.pointAngles(90)
    
    def pointAngles(self,a=0):
        """
        :param aPlus: adds this value to angles.  
            if 0, returns angles in the drawing direction (good for brush strokes)
            if 90, return the outside normals
            if -90 return the inside normals
        
        returns [angle,point]
        """
        mx=len(self.borderLoop)-1
        for i,point in enumerate(borderLoop):
            if i>0:
                prev=borderLoop[i-1]
                if i>=mx:
                    next=borderLoop[0]
                else:
                    next=borderLoop[i+1]
                a2=atan2(point.x-next.x,point.y-next.x)
            else:
                prev=borderLoop[mx]
                next=borderLoop[1]
                a1=atan2(prev.x-point.x,prev.y-point.y)
                a2=atan2(point.x-next.x,point.y-next.x)
            ret.append(((a1+a2)/2+v,point))
            a1=a2 # move it down one so we don't have to recalculate the next one
        return ret
        
    @property
    def centroid(self):
        """
        center of the area
        """
        raise NotImplementedError()
        
    @property
    def pixelArea(self):
        """
        number of pixels in this area
        """
        raise NotImplementedError()
        
    def distanceTo(self,anotherHole):
        """
        get the distance, in pixels, to the nearest edge of another hole
        """
        raise NotImplementedError()
        
    def shrink(self,numPixels):
        """
        shrink the hole
        """
        self.grow(-numPixels)
        
    def grow(self,numPixels):
        """
        grow the hole
        """
        raise NotImplementedError()
        
    def toDistanceMap(self,inside=True,outside=False,bounds=None):
        """
        Get the distance from each point to the border
        You can get the inside, outside, or both
        (default is inside only)
        
        :param inside: get the distances inside the selecion
        :param outside: get the distances outside the selection
        :param bounds: for outside distances, you probably need to set this to
            tell it how much outside area you want!
            
        NOTE:
            see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html
        
        :return: numpy array of distances the size of self.boundingBox 
        """
        order=2
        if bounds is None:
            mask=self.pointsInside
        else:
            if not self.boundingBox.inside(bounds):
                raise Exception('Selection of bounds %s is not within requested bounds %s'%(self.bounds,bounds))
            mask=np.zeroes((bounds.w,bounds.h))
            pInside=self.pointsInside
            mask[bounds.x-self.x:self.x+pInside.shape()[0],bounds.y-self.y:self.y+pInside.shape()[1]]=pInside 
        if inside:
            dist=scipy.ndimage.distance_transform_edt(mask,order)
            if outside:
                mask=1-mask # invert the mask
                dist+=scipy.ndimage.distance_transform_edt(mask,order)
        elif outside:
            mask=1-mask # invert the mask
            dist=scipy.ndimage.distance_transform_edt(mask,order)
        return dist
        
    def toHeightMap(self):
        """
        Get a height map gradient of this hole (values from 0..1) based upon distance to center
        where pixel at self.centroid is 1 and pixels at self.boundingRadius are 0
        
       :return: numpy grayscale image the size of self.boundingBox 
        """
        return np.linalg.norm(self.toDistanceMap(inside=True,outside=False))
        
    def toNormalMap(self):
        """
        Get a normal map of this hole (values from 0..1 in each direction) based upon distance to center
        where pixel at self.centroid is 1 and pixels at self.boundingRadius are 0
        
        The resulting image is rgb where r=x,g=y,b=z and the size of self.boundingBox 
        """
        bb=self.boundingBox
        img=numpy.ndarray(bb[0],bb[1],3)
        r=self.boundingRadius
        c=self.centroid
        for point in self.pointsInside:
            z=point.distanceTo(c)/r
            a=atan2(point.x-c.x,point.y-c.y)
            if a>=180:
                x=1-a/180.0
            else:
                x=(a-180)/180.0
            a-=90;
            if a>=180:
                y=1-a/180.0
            else:
                y=(a-180)/180.0
            img[point.x,point.y]=[x,y,z]
        return img
        
    @property
    def pointsInside(self):
        """
        get all of the points inside this selection
        
        note:
            There are two main ways people do this, winding or crossing. (And people fight about which is "better")
                http://geomalgorithms.com/a03-_inclusion.html
            This uses the crossing algorithm, which is based upon the idea that we start outside, 
            and every time we cross a line we toggle back and forth from outside->inside, or inside->outside
        
        source:
            https://stackoverflow.com/questions/217578/how-can-i-determine-whether-a-2d-point-is-within-a-polygon?rq=1
        
        TODO: this can be a lengthy operation, so cache it
        
        returns [(x,y)] of all points inside
        """
        # Arrays containing the x- and y-coordinates of the polygon's vertices.
        vertx = [point[0] for point in self.points]
        verty = [point[1] for point in self.points]
        # Number of vertices in the polygon
        nvert = len(self.points)
        # Points that are inside
        points_inside = []
        # For every candidate position within the bounding box
        for idx, pos in enumerate(bounding_box_positions):
            # count the number of border crossings
            testx, testy = (pos[0], pos[1])
            c = 0
            for i in range(0, nvert):
                j = i - 1 if i != 0 else nvert - 1
                if( ((verty[i] > testy ) != (verty[j] > testy))   and
                        (testx < (vertx[j] - vertx[i]) * (testy - verty[i]) / (verty[j] - verty[i]) + vertx[i]) ):
                    c += 1
            # If odd, that means that we are inside the polygon
            if c % 2 == 1: 
                points_inside.append(pos)
        return points_inside
        
    def unobscured(self,angle):
        """
        get all lines that are visible from a given angle, eg,
        A
            --a--
            |   |
            c   b
            |   |
            --d--
        From the angle A, only the faces a and c would be visible
        """
        raise NotImplementedError()
        
    def __cmp__(self,other):
        """
        compare this to another object >,<,= are based upon
        area in pixels
        """
        if not isninstance(other,(int,float)):
            other=other.pixelArea
        a=self.pixelArea
        if a>other:
            return 1
        if a<other:
            return -1
        return 0
        
        
class Holes:
    """
    a set of holes, mainly used to encapsulate actions that can operate
    upon a set of holes
    """
    def __init__(self,holes=None):
        if holes is None:
            holes=[]
        self.holes=holes
        
    def nearest(self,toHole,howMany=1):
        """
        get the hole(s) nearest to a given hole
        """
        if howMany<=0 or howMany>len(self.holes):
            howMany=len(self.holes)
        distList=self.distancesFrom(toHole)
        distList.sort()
        distList=distList[0:howMany-1]
        distList=[h for _,h in distList]
        return Holes(distList)
        
    def hull(self):
        """
        return a new hole representing the convex hull of all holes.
        """
        raise NotImplementedError()
        
    def distancesFrom(self,toHole):
        """
        get a list[(dist,hole)] to all holes
        """
        ret=[]
        for hole in self.holes:
            ret.append((hole.distanceTo(toHole),toHole))
        return ret
        
    def under(self,pixelArea):
        """
        get all holes under a given area (in pixels)
        """
        ret=[]
        for hole in self.holes:
            if hole.pixelArea<pixelArea:
                ret.append(hole)
        return Holes(ret)
        
    def over(self,pixelArea):
        """
        get all holes over a given area (in pixels)
        """
        if not isninstance(pixelArea,(int,float):
            pixelArea=pixelArea.pixelArea
        ret=[]
        for hole in self.holes:
            if hole.pixelArea<pixelArea:
                ret.append(hole)
        return Holes(ret)
        
    def __len__(self):
        """
        access this like a [Hole]
        """
        return len(self.holes)
        
    def __getitem__(self,slice):
        """
        access this like a [Hole]
        
        NOTE: returns a Holes() when it would return a list[Hole]
        """
        ret=self.holes[slice]
        if isinstance(ret,Hole):
            return ret
        return Holes(ret)
        
    def __cmp__(self,other):
        """
        compare this to another object >,<,= are based upon
        area in pixels
        """
        if not isninstance(other,(int,float)):
            other=other.pixelArea
        a=self.pixelArea
        if a>other:
            return 1
        if a<other:
            return -1
        return 0
        
    @property
    def pixelArea(self):
        """
        total pixel area of all holes
        """
        a=0
        for h in self.holes:
            a+=h.pixelArea
        return a


def cmdline(args):
    """
    Run the command line
    
    :param args: command line arguments (WITHOUT the filename)
    """
    printhelp=False
    if len(args)<1:
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
        print('  swissCheese.py [options]')
        print('Options:')
        print('   NONE')
        return -1
    return 0


if __name__=='__main__':
    import sys
    sys.exit(cmdline(sys.argv[1:]))

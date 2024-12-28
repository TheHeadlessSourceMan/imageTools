"""
A rectangular bounds representation with lots of unnecessary "cleverness".
"""
import typing
import math


class Bounds:
    """
    A rectangular bounds representation with lots of unnecessary "cleverness".
    """

    def __init__(self,*vals):
        self.x=None
        self.y=None
        self.w=None
        self.h=None
        self.assign(vals)

    def clear(self):
        """
        clear the bounds
        """
        self.x=None
        self.y=None
        self.w=None
        self.h=None

    def __repr__(self):
        return str(self.points)

    def __len__(self):
        return len(self.points)

    def __getitem__(self,key):
        return self.points[key]

    def __iter__(self):
        return self.points.__iter__()

    def assign(self,points):
        """
        assign the points to this object
        """
        self.clear()
        self.maximize(points)

    def copyBounds(self):
        """
        create a copy of these bounds
        """
        return Bounds(self)

    def maximize(self,
        moreBounds:typing.Union["Bounds",typing.Iterable["Bounds"]]
        )->None:
        """
        expands these bounds to encompass moreBounds

        NOTE: if you don't want this object modified, use copyBounds() first
        """
        if hasattr(moreBounds,'__iter__'):
            if moreBounds and isinstance(moreBounds[0],(int,float)):
                for i,val in enumerate(moreBounds): # array of x,y points
                    if i%2==0:
                        if self.x is None or val<self.x:
                            self.x=val
                        if self.x2 is None or val>self.x2:
                            self.x2=val
                    else:
                        if self.y is None or val<self.y:
                            self.y=val
                        if self.y2 is None or val>self.y2:
                            self.y2=val
            else:
                for bounds in moreBounds:
                    self.maximize(bounds)
        else:
            if moreBounds.x<self.x:
                self.x=moreBounds.x
            if moreBounds.x2>self.x2:
                self.x2=moreBounds.x2
            if moreBounds.y<self.y:
                self.y=moreBounds.y
            if moreBounds.y2>self.y2:
                self.y2=moreBounds.y2

    @property
    def x2(self)->float:
        """
        get the right x location
        """
        if self.x is None or self.w is None:
            return None
        return self.x+self.w
    @property
    def y2(self)->float:
        """
        get the bottom y location
        """
        if self.y is None or self.h is None:
            return None
        return self.y+self.h
    @x2.setter
    def x2(self,x2:float):
        """
        set the right x location
        """
        self.w=x2-self.x
    @y2.setter
    def y2(self,y2:float):
        """
        set the bottom y location
        """
        self.h=y2-self.y

    @property
    def points(self
        )->typing.Iterable[typing.Tuple[float,float]]:
        """
        get the object's corner points
        """
        x=self.x
        y=self.y
        x2=self.x2
        y2=self.y2
        return ((x,y),(x2,y),(x2,y2),(x,y2))
    @points.setter
    def points(self,
        points:typing.Iterable[typing.Tuple[float,float]]):
        """
        set the object's corner points
        """
        self.assign(points)

    @property
    def location(self)->typing.Tuple[float,float]:
        """
        get the object's location tuple
        """
        return (self.x,self.y)
    @location.setter
    def location(self,location:typing.Tuple[float,float]):
        """
        set the object's location tuple
        """
        self.move(location,True)

    @property
    def size(self)->typing.Tuple[float,float]:
        """
        get the object's size tuple
        """
        return (self.w,self.h)
    @size.setter
    def size(self,size:typing.Tuple[float,float]):
        """
        set the object's size tuple
        """
        self.w,self.h=size

    @property
    def offset(self)->typing.Tuple[float,float]:
        """
        get the object's offset
        """
        return (self.x,self.y)
    @offset.setter
    def offset(self,offset:typing.Tuple[float,float]):
        """
        set the object's offset
        """
        self.move(offset)

    @property
    def center(self)->typing.Tuple[float,float]:
        """
        get the object's center point
        """
        return (self.x+self.w/2,self.y+self.h/2)
    @center.setter
    def center(self,center:typing.Tuple[float,float]):
        """
        set the object's center point
        """
        self.x=center[0]+self.w/2
        self.y=center[0]+self.y/2

    def __contains__(self,
        points:typing.Iterable[typing.Tuple[float,float]]
        )->bool:
        """
        so you can do something like
            if (10,10) in bounds:
        """
        return self.pointTest(points)
    def pointTest(self,
        points:typing.Iterable[typing.Tuple[float,float]]
        )->bool:
        """
        check to see if any point is within these bounds

        point can be a single point or a list of points
        """
        if isinstance(points,(list,tuple)):
            for point in points:
                if self.pointTest(point):
                    return True
            return False
        return points[0]>=self.x \
            and points[0]<=self.x2 \
            and points[1]>=self.y \
            and points[1]<=self.y2

    def isEnclosedBy(self,
        otherBounds:"Bounds"
        )->bool:
        """
        check to see if we are totally enclosed by otherBounds
        """
        if not isinstance(otherBounds,Bounds):
            otherBounds=Bounds(otherBounds)
        return otherBounds.encloses(self)

    def encloses(self,
        otherBounds:"Bounds"
        )->bool:
        """
        check to see if we totally enclose otherBounds
        """
        if not isinstance(otherBounds,Bounds):
            otherBounds=Bounds(otherBounds)
        return otherBounds.x>=self.x \
            and otherBounds.x2<=self.x2 \
            and otherBounds.y>=self.y \
            and otherBounds.y2<=self.y2

    def overlaps(self,otherBounds:"Bounds")->bool:
        """
        check to see if any point of these bounds overlaps otherBounds
        """
        if not isinstance(otherBounds,Bounds):
            otherBounds=Bounds(otherBounds)
        return otherBounds.pointTest(self.points)

    def move(self,
        location:typing.Tuple[float,float],
        absolute:bool=False
        )->None:
        """
        move bounds to a new location
        """
        if absolute:
            location=(self.x-location[0])
            location=(self.y-location[1])
        self.x+=location[0]
        self.y+=location[1]
        self.x2+=location[0]
        self.y2+=location[1]

    def pivot(self)->None:
        """
        flip these bounds 90 degrees, by swapping x and y values
        """
        self.y,self.x=(self.x,self.y)
        self.h,self.w=(self.w,self.h)

    @classmethod
    def findCenter(cls,
        points:typing.Iterable[typing.Tuple[float,float]]
        )->typing.Tuple[float,float]:
        """
        Utility to find the geographic center of a group of points tuples
        """
        return Bounds(points).center

    @classmethod
    def rotatedPoints(cls,
        points:typing.Iterable[typing.Tuple[float,float]],
        angle:float,
        center:typing.Optional[typing.Tuple[float,float]]=None
        )->typing.Iterable[typing.Tuple[float,float]]:
        """
        Utility to return a set of point tuples based on an existing set of
        point tuples, only rotated.

        If center=None, will findCenter(points) first
        """
        if center is None:
            center=cls.findCenter(points)
        ret=[]
        angle=math.radians(angle)
        sin_a,cos_a=abs(math.sin(angle)),abs(math.cos(angle))
        for point in points:
            ret.append((
                (point[0]-center[0])*cos_a+(point[1]-center[1])*sin_a,
                (point[0]-center[0])*sin_a+(point[1]-center[1])*cos_a
                ))
        return ret

    def rotateFit(self,
        angle:float
        )->None:
        """
        calculate the new bounds to fit a rotated version of these bounds

        angle - in degrees

        NOTE: The algorithm is to rotate each corner point about a given
            center, then create new corner points based on
            min/max x and y location
        """
        if self.w<=0 \
            or self.h<=0 \
            or angle%180==0: # nothing will change, so save some math
            #
            return
        if angle%90==0: # this is a simple x/y value swap
            self.pivot()
        self.assign(Bounds.rotatedPoints(self.points,angle,self.center))

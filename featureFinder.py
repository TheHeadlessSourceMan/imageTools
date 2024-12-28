#!/usr/bin/env
# -*- coding: utf-8 -*-
"""
Find important features in an image.

For more advanced detection (eg faces) requires OpenCV
"""
import os
try:
    # first try to use bohrium, since it could help us accelerate
    # https://bohrium.readthedocs.io/users/python/
    import bohrium as np # type: ignore
except ImportError:
    # if not, plain old numpy is good enough
    import numpy as np
try:
    import cv2 # type: ignore
    has_opencv=True
except ImportError:
    has_opencv=False
from PIL import Image,ImageDraw
from .selectionsAndPaths import selectByColor
from .helper_routines import numpyArray
from . import resizing
from . import colorSpaces


def selectHaar(image,cascadeXmlPath,asMask=True,roundSelection=False):
    """
    Find features in an image using a haar cascade

    :param image: the image to select
    :param cascadeXmlPath: path to the haar cascade xml file to apply
    :param asMask: if True, return the selections as a b&w mask.
        if False, as an array of bounds tuples
    :param roundSelection: if True, draw the feature in mask as an ellipse,
        otherwise as a rectangle

    :return: b&w image where white=interesting, black=uninteresting
        or an (x,y,w,h) array of detected features

    see also:
        https://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html
        https://shahsparx.me/opencv-eye-detection-glasses-opencv/
    """
    if not has_opencv:
        print("WARN: OpenCV not installed.  Facial detection not available!")
        return None
    image=(colorSpaces.grayscale(image)*256).astype(np.uint8)
    haar=cv2.CascadeClassifier(cascadeXmlPath)
    try:
        items=haar.detectMultiScale(image,1.3,5)
    except cv2.error as e:
        searchStr="!empty() in function 'cv::CascadeClassifier::detectMultiScale" # noqa: E501 # pylint: disable=line-too-long
        if str(e).find(searchStr)>=0:
            msg=f'XML cascade filter not found "{cascadeXmlPath}"'
            raise Exception(msg) from e
        raise e
    if not asMask:
        return items
    return boundsToMask(resizing.getSize(image),items,roundSelection)


def selectEyes(image,asMask=True):
    """
    Find eyes in an image.

    :param image: the image to select
    :param asMask: if True, return the selections as a b&w mask.
        if False, as an array of bounds tuples

    :return: b&w image where white=interesting, black=uninteresting
        or an (x,y,w,h) array of detected features
    """
    basedir=os.sep.join((
        os.path.abspath(__file__).rsplit(os.sep,2)[0],
        'cv_data',
        'haarcascades',
        ''))
    eye_detector=basedir+'haarcascade_eye.xml'
    return selectHaar(image,eye_detector,asMask,roundSelection=True)


def boundsToMask(imgSize,bounds,roundSelection=False):
    """
    Turn one or more bounds specifications into a black and white selection.

    :param imgSize: size of the new mask image
    :param bounds: an x,y,w,h bounds, or an array of them
    :param roundSelection: whether to use rectangles or ellipses as selections
    """
    outImage=Image.new('L',(imgSize[0],imgSize[1]),0)
    draw=ImageDraw.Draw(outImage)
    if bounds:
        if not hasattr(bounds[0],'__getitem__'):
            bounds=(bounds)
        if roundSelection:
            print(len(bounds))
            for b in bounds:
                (x,y,w,h)=b
                draw.ellipse((x,y,x+w,y+h),fill=255)
        else:
            for b in bounds:
                (x,y,w,h)=b
                draw.rectangle((x,y,x+w,y+h),fill=255)
    return numpyArray(outImage)


def selectFaces(image,asMask=True):
    """
    Find human faces in an image.

    :param image: the image to select
    :param asMask: if True, return the selections as a b&w mask.
        if False, as an array of bounds tuples

    :return: b&w image where white=interesting, black=uninteresting
        or an (x,y,w,h) array of detected features
    """
    basedir=os.sep.join((
        os.path.abspath(__file__).rsplit(os.sep,2)[0],
        'cv_data',
        'haarcascades',
        ''))
    face_detector=basedir+'haarcascade_frontalface_default.xml'
    return selectHaar(image,face_detector,asMask,round=True)


def selectHair(image):
    """
    Find hair and return a mask.

    https://stackoverflow.com/questions/4822999/detecting-hair-in-a-portrait-image
    """
    raise NotImplementedError()
    if not has_opencv:
        print("WARN: OpenCV not installed.  Facial detection not available!")
        return None
    outImage=Image.new('L',image.size,0)
    draw=ImageDraw.Draw(outImage)
    basedir=os.sep.join((
        os.path.abspath(__file__).rsplit(os.sep,1)[0],
        'cv_data',
        'haarcascades',
        ''))
    return outImage


def selectBackground(image):
    """
    Detect image background and return a mask.

    https://stackoverflow.com/questions/19221877/opencv-how-to-use-createbackgroundsubtractormog
    """
    raise NotImplementedError()
    if not has_opencv:
        print("WARN: OpenCV not installed.  Facial detection not available!")
        return None
    outImage=Image.new('L',image.size,0)
    draw=ImageDraw.Draw(outImage)
    basedir=os.sep.join((
        os.path.abspath(__file__).rsplit(os.sep,1)[0],
        'cv_data',
        'haarcascades',
        ''))
    return outImage


def selectSkin(image):
    """
    Find human skin by color and return a mask.

    This is a match on HS colorspace, which, against popular rumor,
    performs better than other color spaces.
        "Color Space for Skin Detection - A Review"
            by Nikhil Rasiwasia,
            Fondazione Graphitech,
            University of Trento, (TN) Italy
        "Comparison  of  Five  Color  Models  in  Skin  Pixel  Classification"
            by Benjamin D. Zarit, Boaz J. Super, Francis K. H. Quek
            Electrical Engineering and Computer Science
            University of Illinois at Chicago
        http://answers.opencv.org/question/3300/skin-detection/

    :param image: image to match
    :return: a black and white mask
    """
    return selectByColor(image,((0,48,80),(20,255,255)),colorspace='HSV')

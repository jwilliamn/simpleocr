#!/usr/bin/env python
# coding: utf-8

"""
    mindisApp
    ============================

    File rename of DJs
    Structure/
        core

    _copyright_ = 'Copyright (c) 2017 J.W.', see AUTHORS for more details
    _license_ = GNU General Public License, see LICENSE for more details
"""

import numpy as np
import cv2
import sys
import os
import time

import random
import math
import json

from PIL import Image, ImageDraw
from collections import defaultdict
from matplotlib import pyplot as plt
from os.path import join
from scipy.ndimage.filters import rank_filter
from pytesseract import image_to_string


# Input  settings ####
# APP_DIR = path/to/parentPath
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Function definitions ####

def crop_image(img , x, y, w, h):
    """Remove everything outside the given coordinates"""
    # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
    #cropIm = img[0:y, 0:int(img.shape[1]/2)]
    cropedIm = img[y: y + h, x: x + w]

    #plt.imshow(cropedIm, cmap=plt.cm.gray)
    #plt.title("Croped image")
    #plt.show()

    return cropedIm

def find_border_components(contours, img, targetContour, stage=1):
    borders = []
    area = img.shape[0] * img.shape[1]

    if stage == 2 or stage == 3:
        # Stright Bounding Rectangle ####
        x,y,w,h = cv2.boundingRect(targetContour)
        borders.append((1, x, y, w, h))
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        print('Stages: x & y - w & h', x, y, w, h)
    else:
        for i, c in enumerate(contours):
            x,y,w,h = cv2.boundingRect(c)

            #borders.append((i, x, y, w, h))
            #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            if w * h > 0.5 * area:
                borders.append((i, x, y, w, h))
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)            

    print('borders', type(borders), borders)
    
    #plt.imshow(img)
    #plt.title('Rectangles of target contours')
    #plt.show()
    return borders


def dilate(img, iterations=1, N=5): 
    """Dilate using an NxN '+' sign shape. img is np.uint8."""
    #kernel = np.zeros((N,N), dtype=np.uint8)
    #kernel[(N-1)/2,:] = 1
    #dilated_image = cv2.dilate(ary / 255, kernel, iterations=iterations)
    #kernel = np.zeros((N,N), dtype=np.uint8)
    #kernel[:,(N-1)/2] = 1
    #dilated_image = cv2.dilate(dilated_image, kernel, iterations=iterations)

    kernel = np.ones((N,N), dtype=np.uint8)
    dilated_image = cv2.dilate(img, kernel, iterations=iterations)
    #plt.imshow(dilated_image)
    #plt.show()
    print('Dilated image', type(dilated_image))
    return dilated_image

def closingMor(img, N=5): 
    """Close using an NxN '+' sign shape. img is np.uint8."""

    kernel = np.ones((N,N), dtype=np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    #plt.imshow(closing)
    #plt.show()
    print('Closing image', type(closing))
    return closing

def find_components(edges, Ndil, Nclos, dilIter = 1):
    """Dilate the image until there are just a few connected components.
    Returns contours for these components."""
    # Perform increasingly aggressive dilation until there are just a few
    # connected components.

    # Morphological transformations
    dilation = dilate(edges, dilIter, Ndil)
    closing = closingMor(dilation, Nclos)
    
    # TODO: dilate image _before_ finding a border. This is crazy sensitive!
    ret, thresh = cv2.threshold(closing,127,255,0)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours

def contour_max_area(contours):
    area = []
    for c in range(0,len(contours)):
        cnt = contours[c]
        M = cv2.moments(cnt)
        area.append(cv2.contourArea(cnt))

    area = np.array(area)
    maxA = area.argmax()
    print('contour of max Area', maxA)
    return maxA

def aprox_contours(cnts):
    # Contour perimeter ####
    perimeter = cv2.arcLength(cnt, True)

    # Contour aproximation ####
    epsilon = 0.1*cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    squareCnt = None
    # loop over our contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if our approximated contour has four points, then
        # we can assume that we have found the big square
        if len(approx) == 4:
            squareCnt = approx
            break

    return squareCnt


def downscale_image(image, max_dim=2048):
    # we need to keep in mind aspect ratio so the image does
    # not look skewed or distorted -- therefore, we calculate
    # the ratio of the new image to the old image
    r = max_dim / image.shape[1]
    dim = (max_dim, int(image.shape[0] * r))

    # perform the actual resizing of the image and show it
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return resized

def process_image(path):
    #img  = cv2.imread(path, 0)
    im = cv2.imread(path)
    img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    print('Original image - type,  h & w', type(img), img.shape[0], img.shape[1])

    #img = downscale_image(img)

    #plt.subplot(121),plt.imshow(img)
    #plt.title('Gray Image with cv2 colorbgr2gray') #, plt.xticks([]), plt.yticks([])
    #plt.subplot(122),plt.imshow(imOg)
    #plt.title('gray Image when reading') #, plt.xticks([]), plt.yticks([])
    #plt.show()

    # Remove noise
    print('Removing noise....')
    xMarg = int(200*img.shape[1]/2592)
    mask = [[0,0],[0,img.shape[0]],[xMarg,img.shape[0]],[xMarg,0],[0,0]]
    mask = np.array(mask)
    #print('mask', type(mask), mask)
    cv2.drawContours(img, [mask], 0, (255,255,255), -1)
    #plt.imshow(img, cmap=plt.cm.gray)
    #plt.title('Noise removed')
    #plt.show()

    print('Detecting edges')
    edges = cv2.Canny(img, 100, 200)
    #plt.imshow(edges, cmap=plt.cm.gray)
    #plt.title('Edges')
    #plt.show()

    # Find components
    print('Finding contours')
    contours = find_components(edges, Ndil=35, Nclos=35)
    print('Total contours found', len(contours), type(contours))

    cnts = sorted(contours, key = cv2.contourArea, reverse = True)
    
    # Get contour of max area
    maxA = contour_max_area(contours)
    cnt = contours[maxA]

    
    # Important ####
    imC = im.copy()
    #cv2.drawContours(imC, [cnt], 0, (255,255,255), -1)
    #cv2.drawContours(imC, contours, -1, (0,255,0), 5)
    #plt.imshow(imC)
    #plt.show()

    #squareCnt = aprox_contours(cnts)
    #cv2.drawContours(imgC, [squareCnt], -1, (0, 255, 0), 3)
    #plt.imshow(imgC)
    #plt.show()

    # Rectangles of contours
    borders = find_border_components(contours, imC, targetContour = cnt)
    y = borders[0][2]   # y coordinate of big box

    # Crop picture
    cropIm = crop_image(img, x=0, y=0, w=int(img.shape[1]/2) , h=y)



    # Working on croped image Stage 2
    print('Working on croped image stage 2')
    edgesA = cv2.Canny(cropIm, 100, 200)
    #plt.imshow(edgesA, cmap=plt.cm.gray)
    #plt.title('Edges Aft')
    #plt.show()
    
    # Find components
    contours = find_components(edgesA, Ndil=30, Nclos=35)
    print('Total contours found', len(contours), type(contours))

    cnts = sorted(contours, key = cv2.contourArea, reverse = True)
    
    # Get contour of max area
    maxA = contour_max_area(contours)
    cnt = contours[maxA]

    # Rectangles of contours
    borders = find_border_components(contours, cropIm.copy(), targetContour = cnt, stage = 2)
    x = borders[0][1]
    y = borders[0][2]
    w = borders[0][3]
    h = borders[0][4]

    # Crop picture
    cropIm = crop_image(cropIm, x, y, w, h)



    # Working on croped image Stage 3
    print('Working on croped image stage 3')
    edgesA = cv2.Canny(cropIm, 100, 200)
    
    # Find components
    contours = find_components(edgesA, Ndil=25, Nclos=5)
    print('Total contours found', len(contours), type(contours))

    cnts = sorted(contours, key = cv2.contourArea, reverse = False)
    
    # Get contour of min area
    cnt = cnts[0]

    # Rectangles of contours
    borders = find_border_components(contours, cropIm.copy(), targetContour = cnt, stage = 3)
    x = borders[0][1]
    y = borders[0][2]
    w = borders[0][3]
    h = borders[0][4]

    # Crop picture
    cropfin = crop_image(cropIm, x, y, w, h)



    # Working on croped image Stage 4
    print('Final stage')
    edgesA = cv2.Canny(cropfin, 100, 200)

    kernel = np.ones((1,1),np.uint8)
    erosion = cv2.erode(cropfin, kernel, iterations = 1)

    #plt.imshow(erosion, cmap=plt.cm.gray)
    #plt.title("final crop eroded")
    #plt.show()

    pilIm = Image.fromarray(np.rollaxis(erosion,0,0))


    # Recognition ####
    #text = image_to_string(pilIm, lang='eng')
    text = image_to_string(pilIm, lang='spa')
    print('Recognized register', type(text), text)

    #Save final croped picture
    cv2.imwrite('test/croped_' + text + '.png',cropfin)

    if len(text) != 0:
        return text
    else:
        return None
 
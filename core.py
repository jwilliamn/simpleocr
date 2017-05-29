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

def find_components(edges, max_components=16):
    """Dilate the image until there are just a few connected components.
    Returns contours for these components."""
    # Perform increasingly aggressive dilation until there are just a few
    # connected components.
    count = 21
    dilation = 5
    n = 1
    while count > 16:
        n += 1
        dilated_image = dilate(edges, N=3, iterations=n)
        contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        count = len(contours)
    #print dilation
    #Image.fromarray(edges).show()
    #Image.fromarray(255 * dilated_image).show()
    return contours

def remove_border(contour, ary):
    """Remove everything outside a border contour."""
    # Use a rotated rectangle (should be a good approximation of a border).
    # If it's far from a right angle, it's probably two sides of a border and
    # we should use the bounding box instead.
    c_im = np.zeros(ary.shape)
    r = cv2.minAreaRect(contour)
    degs = r[2]
    if angle_from_right(degs) <= 10.0:
        box = cv2.cv.BoxPoints(r)
        box = np.int0(box)
        cv2.drawContours(c_im, [box], 0, 255, -1)
        cv2.drawContours(c_im, [box], 0, 0, 4)
    else:
        x1, y1, x2, y2 = cv2.boundingRect(contour)
        cv2.rectangle(c_im, (x1, y1), (x2, y2), 255, -1)
        cv2.rectangle(c_im, (x1, y1), (x2, y2), 0, 4)

    return np.minimum(c_im, ary)

def find_border_components(contours, ary):
    borders = []
    area = ary.shape[0] * ary.shape[1]
    for i, c in enumerate(contours):
        x,y,w,h = cv2.boundingRect(c)
        if w * h > 0.5 * area:
            borders.append((i, x, y, x + w - 1, y + h - 1))
    return borders

def dilate(img, N=35, iterations): 
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
    print(type(dilated_image))
    return dilated_image

def closing(img, N=35): 
    """Close using an NxN '+' sign shape. img is np.uint8."""

    kernel = np.ones((N,N), dtype=np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    #plt.imshow(closing)
    #plt.show()
    print(type(closing))
    return closing

def morphologicalOp(img)
    edges = cv2.Canny(img, 100, 200)
    #plt.imshow(edges, cmap=plt.cm.gray)
    #plt.show()

    dilation = dilate(img, interations=1)
    closing = closing(dilation)
    return closing


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
    img  = cv2.imread(path, 0)
    #img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    print(type(img), img.shape[0], img.shape[1])

    #img = downscale_image(img)

    #plt.subplot(121),plt.imshow(img)
    #plt.title('Gray Image with cv2 colorbgr2gray') #, plt.xticks([]), plt.yticks([])
    #plt.subplot(122),plt.imshow(imOg)
    #plt.title('gray Image when reading') #, plt.xticks([]), plt.yticks([])
    #plt.show()
    
    # Morphological transformations
    imgMorph = morphologicalOp(img) # result of image after some operations


    # TODO: dilate image _before_ finding a border. This is crazy sensitive!
    ret, thresh = cv2.threshold(closing,127,255,0)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cnts = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
    
    print('contours', len(contours), type(contours))
    #print('contours', len(cnts), type(cnts))
    area = []
    for c in range(0,len(contours)):
        cnt = contours[c]
        M = cv2.moments(cnt)
        #print(M)

        area.append(cv2.contourArea(cnt))

    #print('area', type(area), area)
    area = np.array(area)
    maxA = area.argmax()
    print('max Area',maxA)
    #cnts = sorted(area, reverse=True)

    cnt = contours[maxA]

    # Important ####
    #cv2.drawContours(im, [cnt], 0, (255,255,255), -1)
    #cv2.drawContours(im, contours, -1, (0,255,0), 5)
    #plt.imshow(im)
    #plt.show()


    # Contour perimeter ####
    perimeter = cv2.arcLength(cnt, True)

    # Contour aproximation ####
    epsilon = 0.1*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)

    screenCnt = None
    # loop over our contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if our approximated contour has four points, then
        # we can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break

    #cv2.drawContours(im, [screenCnt], -1, (0, 255, 0), 3)
    #plt.imshow(im)
    #plt.show()

    # Stright Bounding Rectangle ####
    x,y,w,h = cv2.boundingRect(cnt)
    imCop = im.copy()
    cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)

    print('x & y', x, y)
    plt.imshow(im)
    plt.show()

    # Crop picture
    # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
    #cropIm = im[0:y-10, 0:int(img.shape[1]/2)]
    cropIm = imCop[0:y, 0:int(img.shape[1]/2)]

    plt.imshow(cropIm)
    plt.title("After Largest box croped")
    plt.show()

    # Working on croped image
    edgesAft = cv2.Canny(cropIm, 100, 200)

    plt.imshow(edgesAft, cmap=plt.cm.gray)
    plt.show()
    
    kernel = np.ones((30,30),np.uint8)
    dilation = cv2.dilate(edgesAft, kernel, iterations = 1)
    kernel = np.ones((35,35),np.uint8)
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)

    plt.imshow(closing)
    plt.show()

    # find contours
    ret, thresh = cv2.threshold(closing,127,255,0)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cnts = sorted(contours, key = cv2.contourArea, reverse = True)
    
    print('new len conts', len(contours))
    #print('new conts', cnts)

    #cv2.drawContours(cropIm, [cnts[0]], 0, (0,255,0), 3)
    #cv2.drawContours(cropIm, contours, -1, (0,255,0), 5)
    #plt.imshow(cropIm)
    #plt.show()   

    
    # Stright Bounding Rectangle ####
    x,y,w,h = cv2.boundingRect(cnts[0])
    cv2.rectangle(cropIm.copy(),(x,y),(x+w,y+h),(0,255,0),2)

    print('x & y - w & h', x, y, w, h)
    plt.imshow(cropIm)
    plt.show()

    # Crop picture
    crop2 = cropIm[y:y+h, x:x+w]
    crop2Cop = crop2.copy()

    plt.imshow(crop2)
    plt.show()


    # Stage 3
    # Working on croped image
    print('Stage 3')
    edgesAft = cv2.Canny(crop2, 100, 200)

    plt.imshow(edgesAft, cmap=plt.cm.gray)
    plt.show()
    
    kernel = np.ones((25,25),np.uint8)
    dilation = cv2.dilate(edgesAft, kernel, iterations = 1)
    kernel = np.ones((5,5),np.uint8)
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)

    plt.imshow(closing)
    plt.show()

    # find contours
    ret, thresh = cv2.threshold(closing,127,255,0)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cnts = sorted(contours, key = cv2.contourArea, reverse = False)
    
    print('new len conts', len(contours))
    
    # Stright Bounding Rectangle ####
    x,y,w,h = cv2.boundingRect(cnts[0])
    cv2.rectangle(crop2,(x,y),(x+w,y+h),(0,255,0),2)

    print('x & y - w & h', x, y, w, h)
    plt.imshow(crop2)
    plt.show()

    # Crop picture
    cropfin = crop2Cop[y:y+h, x:x+w]

    plt.imshow(cropfin)
    plt.title("Final image redy to be predicted")
    plt.show()

    print('Final Stage')
    edgesAft = cv2.Canny(cropfin, 100, 200)
    kernel = np.ones((1,1),np.uint8)
    #kernel = np.zeros((1,1),np.uint8)
    erosion = cv2.erode(cropfin, kernel, iterations = 1)
    #closing = cv2.morphologyEx(cropfin, cv2.MORPH_CLOSE, kernel)

    plt.imshow(erosion)
    #plt.imshow(closing)
    plt.title("final crop eroded")
    plt.show()

    pilIm = Image.fromarray(np.rollaxis(erosion,0,0))
    #pilIm = Image.fromarray(np.rollaxis(closing,0,0))

    #pilIm = Image.fromarray(np.rollaxis(cropfin,0,0))

    # Recognition
    #text = image_to_string(pilIm, lang='eng')
    text = image_to_string(pilIm, lang='spa')
    print('Recognized register', text)

    #Save final croped picture
    cv2.imwrite('test/croped_' + text + '.png',cropfin)


    """
    border_contour = None
    if len(borders):
        border_contour = contours[borders[0][0]]
        edges = remove_border(border_contour, edges)

    edges = 255 * (edges > 0).astype(np.uint8)

    # Remove ~1px borders using a rank filter.
    maxed_rows = rank_filter(edges, -4, size=(1, 20))
    maxed_cols = rank_filter(edges, -4, size=(20, 1))
    debordered = np.minimum(np.minimum(edges, maxed_rows), maxed_cols)
    edges = debordered

    contours = find_components(edges)
    if len(contours) == 0:
        print('%s -> (no text!)' % path)
        return

    crop = find_optimal_components_subset(contours, edges)
    crop = pad_crop(crop, contours, edges, border_contour)

    crop = [int(x / scale) for x in crop]  # upscale to the original image size.
    #draw = ImageDraw.Draw(im)
    #c_info = props_for_contours(contours, edges)
    #for c in c_info:
    #    this_crop = c['x1'], c['y1'], c['x2'], c['y2']
    #    draw.rectangle(this_crop, outline='blue')
    #draw.rectangle(crop, outline='red')
    #im.save(out_path)
    #draw.text((50, 50), path, fill='red')
    #orig_im.save(out_path)
    #im.show()
    text_im = orig_im.crop(crop)
    text_im.save(out_path)
    print('%s -> %s' % (path, out_path))
    """
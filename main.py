#!/usr/bin/env python
# coding: utf-8

"""
    mindisApp
    ============================

    File rename of DJs
    Structure/
        rename

    _copyright_ = 'Copyright (c) 2017 J.W.', see AUTHORS for more details
    _license_ = GNU General Public License, see LICENSE for more details
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
import os
import time

from wand.image import Image
from wand.color import Color
from PyPDF2 import PdfFileWriter, PdfFileReader
from os.path import join

from core import process_image


# Input  settings ####
# APP_DIR = path/to/parentPath
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Function definitions ####
def processPdf(originalPdf):
    inputPdf = PdfFileReader(open(originalPdf, 'rb'))

    pdfname = os.path.basename(originalPdf)
    pdfname = pdfname.split('.')
    pdfname = pdfname[len(pdfname) - 2]

    if not os.path.exists(join(APP_DIR, 'tmp/')):
        os.makedirs(join(APP_DIR, 'tmp/'))

    outputPath = []
    for i in range(0,1): #inputPdf.getNumPages()):
        p = inputPdf.getPage(i)
        outputPdf = PdfFileWriter()
        outputPdf.addPage(p)

        with open(join(APP_DIR, 'tmp/' + pdfname + '_%1d.pdf') % (i + 1), 'wb') as f:
            outputPdf.write(f)
            outputPath.append(f.name)

    outputPath = np.array(outputPath)
    #print('outputPdf', type(outputPath), outputPath)
    return outputPath, inputPdf.getNumPages()


def convert_pdf_png(filePath, numPages):
    imagePath = []
    for i in range(0,1): #numPages):
        path = filePath[i]
        pathName = path.split('.')

        print('Converting page %d' % (i + 1))
        try:
            with Image(filename=path, resolution=300) as img:
                with Image(width=img.width, height=img.height, background=Color('white')) as bg:
                    bg.composite(img, 0, 0)
                    bg.save(filename=pathName[0] + '.png')
        except Exception as e:
            print('Unable to convert pdf file', e)
            raise

        imagePath.append(pathName[0] + '.png')
    
    imagePath = np.array(imagePath)
    return imagePath


# Main function ####
if __name__ == '__main__':
    """ .........
    To run the app, execute the following in terminal:

    [terminal_prompt]$ python rename.py path/to/image.pdf

    Currently the app supports images in the following formats: 
        .png
        .jpeg
        .jpg
        .pdf
    """
    print("Hi there, its mindisApp I'll try to be helpful :) \nBut I'm still just a robot. Sorry!")

    arg = sys.argv[1]
    print('arg', arg)
    splitArg = arg.split('.')


    if splitArg[1] == 'png' or splitArg[1] == 'jpeg' or splitArg[1] == 'jpg':
        print("File is a picture!")
        imgPath = np.array([arg])
    else:
        if splitArg[1] == 'pdf' or splitArg[1] == 'PDF':
            print('File is a pdf!')
            pdfPath, numPag = processPdf(arg)
            imgPath = convert_pdf_png(pdfPath, numPag)

            # print('imgPath__', type(imgPath), imgPath)
        else:
            raise ValueError(splitArg[1] + ' File format cannot be processed :(!')

    #print('imgPath', type(imgPath), len(imgPath))
    print('imgPath', imgPath)

    for path in imgPath:
        out_path = path.replace('.png', '.crop.png')
        if os.path.exists(out_path): continue
        
        try:
            process_image(path, out_path)
        except Exception as e:
            print('%s %s' % (path, e))
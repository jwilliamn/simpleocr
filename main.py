#!/usr/bin/env python
# coding: utf-8

"""
    simpleOcr
    ============================
    The purpose is to recognize the Register number (N° Registro) of scanned documents
    then rename the documents with the recognized numbers to store to a database. 

    Structure:
        simpleOcr/
            core.py
            main.py
            pdfprocess.py
            README.md
            tmp/

    _copyright_ = 'Copyright (c) 2017 J.W.', see AUTHORS for more details
    _license_ = GNU General Public License, see LICENSE for more details
"""

import numpy as np
import cv2
import sys
import os
import time
from matplotlib import pyplot as plt

from wand.image import Image
from wand.color import Color
from PyPDF2 import PdfFileWriter, PdfFileReader
from os.path import join

from core import process_image
from pdfprocess import processPdf, convert_pdf_png


# General settings ####
# APP_DIR = path/to/parentPath
APP_DIR = os.path.dirname(os.path.abspath(__file__))



# Main function ####
if __name__ == '__main__':
    """ ..................
    To run the app, execute the following in terminal:
    [terminal_prompt]$ python main.py path/to/image.pdf
    Currently the app supports images in the following formats: .png, .jpeg, .jpg und .pdf
    """
    print("Hi there, its simpleOcr -  I'll try to be helpful :) \nBut I'm still just a robot. Sorry!")

    # Read the inputs
    arg = sys.argv[1]
    print('sys.arg', arg)
    splitArg = arg.split('.')

    # Handling the type of file
    if splitArg[1] == 'png' or splitArg[1] == 'jpeg' or splitArg[1] == 'jpg':
        print("File is a picture!")
        imgPath = np.array([arg])
    else:
        if splitArg[1] == 'pdf' or splitArg[1] == 'PDF':
            print('File is a pdf!')
            inputPdf, pdfPath, numPag = processPdf(arg)
            imgPath = convert_pdf_png(pdfPath, numPag)
        else:
            raise ValueError(splitArg[1] + ' File format cannot be processed :(!')
    #print('imgPath', type(imgPath), len(imgPath))

    # Actual processing
    for path in imgPath:
        try:
            text = process_image(path)
            continue
        except Exception as e:
            print('Exception %s %s' % (path, e))

    outPdf = PdfFileWriter()
    outPdf.appendPagesFromReader(inputPdf)
    with open(join(APP_DIR, 'output/' + text + '.pdf'), 'wb') as fi:
            outPdf.write(fi)
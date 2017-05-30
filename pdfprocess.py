#!/usr/bin/env python
# coding: utf-8

"""
    simpleOcr.pdfprocess
    ============================
    Processess pdfs, split and convert to images

    Structure:
        pdfprocess.py

    _copyright_ = 'Copyright (c) 2017 J.W.', see AUTHORS for more details
    _license_ = GNU General Public License, see LICENSE for more details
"""

import numpy as np
import os
from os.path import join

from wand.image import Image
from wand.color import Color
from PyPDF2 import PdfFileWriter, PdfFileReader



# General settings ####
# APP_DIR = path/to/parentPath
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Function definitions ####
def processPdf(originalPdf):
    inputPdf = PdfFileReader(open(originalPdf, 'rb'))

    pdfname = os.path.basename(originalPdf)
    pdfname = pdfname.split('.')
    pdfname = pdfname[len(pdfname) - 2]

    if not os.path.exists(join(APP_DIR, 'input/tmp/')):
        os.makedirs(join(APP_DIR, 'input/tmp/'))

    outputPath = []
    for i in range(0,1): #inputPdf.getNumPages()):
        p = inputPdf.getPage(i)
        outputPdf = PdfFileWriter()
        outputPdf.addPage(p)

        with open(join(APP_DIR, 'input/tmp/' + pdfname + '_%1d.pdf') % (i + 1), 'wb') as f:
            outputPdf.write(f)
            outputPath.append(f.name)

    outputPath = np.array(outputPath)
    #print('outputPdf', type(outputPath), outputPath)
    return inputPdf, outputPath, inputPdf.getNumPages()


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

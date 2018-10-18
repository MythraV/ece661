#!/usr/bin/env python3
############################################################################
# Script for homework 6 ECE661
#  Harris corner detector code
#  - Contains functions to perform thresholding using Otsu's method.
#  - Functions for texutre based segmentation and contour extraction
# Author: Mythra Balakuntala
############################################################################

import cv2, numpy as np, math, time, matplotlib
import time


class ImageSegment():
    def __init__(self):
        self.img_pth = {    1:'HW6Pics/lighthouse.jpg',
                            2:'HW6Pics/baby.jpg',
                            3:'HW6Pics/ski.jpg',
                            }
# Otsu threshold function
    def otsu(self, img, ):
        #----------------------------------------------------------
        # Implementation of otsu's method to maximize the inter-class
        # variance.
        # Inputs: img - The image (1-channel)
        # Outputs: Otsu thresholded image
        #-----------------------------------------------------------

# Multi channel Otsu threshold function
    def multi_otsu(self, img, type):
        #----------------------------------------------------------
        # Otsu for multi channel inputs, the channels are combined
        #    based on type.
        # Inputs: img - The image (with any number of channels)
        #   type - 1: If no. of channels > 1 then logical and is used.
        #          2: If no. of channels > 1 iteratively threshold.
        # Outputs: Otsu thresholded image
        # Uses the otsu function to compute thresholded image
        #-----------------------------------------------------------
        if img.shape()
        if type==1:
            for i in range(img.shape[2]):



# Function to view image
    def viewimg(self,img):
        cv2.namedWindow('resultImage',cv2.WINDOW_NORMAL)
        cv2.imshow('resultImage',img)
        cv2.resizeWindow('resultImage', 600,600)
        key = 1
        print('Press q to quit')
        while key !=ord('q'):       # Quit on pressing q
            key = cv2.waitKey(0)
        cv2.destroyAllWindows()

# Main function
    def main(self, img_id):
        #----------------------------------------------------------
        # Calls otsu, texture segmentation and contour extraction
        # Inputs: img_id - Id for the first image
        # Outputs: None ,
        # Writes output images from otsu, texture segmentation and
        # contour extraction to the folder
        #-----------------------------------------------------------
        # Find corner points for image 1
        img_rgb = cv2.imread(self.img_names[img_id])
        img = cv2.cvtColor(img1_rgb,cv2.COLOR_BGR2GRAY)




if __name__ == '__main__':
    imgseg = ImageSegment()
    imgseg.main(1)  # imgseg.main(img id)

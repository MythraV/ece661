#!/usr/bin/env python3
############################################################################
# Script for homework 6 ECE661
# Image Segmentation file
#  - Contains functions to perform thresholding using Otsu's method.
#  - Functions to perform Otsu on multiple channels
#  - Functions for texutre based segmentation
#  - Function for contour extraction
# Author: Mythra Balakuntala
############################################################################

import cv2, numpy as np, math, time
import matplotlib.pyplot as plt
import time


class ImageSegment():
    def __init__(self):
        self.img_pth = {    1:'HW6Pics/lighthouse.jpg',
                            2:'HW6Pics/baby.jpg',
                            3:'HW6Pics/ski.jpg',
                            }
        self.ot_choice = {  1:np.array([[1,1,1],[0,0,1],[1,1,1]]),
                            2:np.array([[0,1,0],[0,0,0],[1,1,1]]),
                            3:np.array([[0,0,0],[0,0,1],[1,1,1]]),
                            }
# Histogram generation function
    def hist(self,img, gbins=256, scale = 255):
        #----------------------------------------------------------
        # Intesity histogram generation function
        # Inputs: img - The image (1-channel), gbins - number of bins
        #   scale - scale of input data (0-255 for gray level image)
        # Outputs: hist_counts - Histogram counts
        #-----------------------------------------------------------
        hist_counts = np.zeros(gbins)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                hist_counts[int(img[i,j]/np.ceil(scale/gbins))] += 1
        return(hist_counts)

# Otsu threshold function
    def otsu(self, img, gbins=256, foreg = 0):
        #----------------------------------------------------------
        # Implementation of otsu's method to maximize the inter-class
        # variance.
        # Inputs: img - The image (1-channel), gbins - number of bins,
        #   foreg - foreground indicator.. 0:lower than thres,
        #    1: greater than thres
        # Outputs: Otsu thresholded image
        #-----------------------------------------------------------
        # Generate histogram bins
        res_img = np.ones(img.shape)
        N = img.shape[0]*img.shape[1]  #total number of pixels
        k= 0
        print('Bins, pixels = ',gbins, N)
        hist_counts = self.hist(img, gbins)/N
        wk = 0
        mk = 0
        mt = sum((np.arange(gbins)+1)*hist_counts)
        max_sb = 0.0;
        for i in range(1,gbins+1):
            wk += hist_counts[i-1]
            mk += i*hist_counts[i-1]
            if(wk == 0 or (1-wk)==0):
                continue
            sb = ((mt*wk - mk)**2)/(wk*(1-wk));
            if ( sb >= max_sb ):
                k = i
                max_sb = sb
        print('Otsu threshold is ',k)
        if foreg:
            res_img[np.where(img<=k)] = 0
        else:
            res_img[np.where(img>=k)] = 0
        return(res_img)

# Multi channel Otsu threshold function
    def multi_otsu(self, img, type=1, fchoice=np.array([])):
        #----------------------------------------------------------
        # Otsu for multi channel inputs, the channels are combined
        #    based on type.
        # Inputs: img - The image (with any number of channels)
        #   type - 1: If no. of channels > 1 then logical and is used.
        #          2: If no. of channels > 1 iteratively threshold.
        # Outputs: Otsu thresholded image
        # Uses the otsu function to compute thresholded image
        #-----------------------------------------------------------
        if fchoice.shape[0] == 0:
            fchoice = np.ones(img.shape[2])
        rec_img = img
        if(len(img.shape)==2):      # If only one channel
            return(self.otsu(img))
        elif type==1:               # Logical and each channel otsu
            res_img = np.ones(img.shape[:2])    # Resultant image
            # Array to store otsu for each channel
            thres_img = np.zeros(img.shape)
            for i in range(img.shape[2]):
                thres_img[:,:,i] = self.otsu(img[:,:,i],foreg = fchoice[i])    # Comput otsu
                # And with result until prev channel
                res_img = thres_img[:,:,i]*res_img
                # self.viewimg(thres_img[:,:,i])      # View each channel result ***
                # cv2.imwrite(('img' + str(3) +'_ot'+str(i+1)+'_rgb.jpg'),(255\
                # *thres_img[:,:,i]).astype(np.uint8))
            return(res_img.astype(np.uint8))
        elif type==2:               # Iteratively apply otsu prev channel result
            # Array to store otsu for each channel
            thres_img = np.zeros(img.shape)
            thres_img[:,:,0] = self.otsu(img[:,:,0],foreg = fchoice[0])  # For first channel
            # self.viewimg(thres_img[:,:,0])
            rec_img = (thres_img[:,:,0])[:,:,np.newaxis]*img
            # self.viewimg(rec_img.astype(np.uint8))
            for i in range(1,img.shape[2]):
                # Compute otsu on next channel based on foreground of prev.
                self.viewimg(thres_img[:,:,i-1]*rec_img[:,:,i])
                thres_img[:,:,i] = self.otsu(thres_img[:,:,i-1]*rec_img[:,:,i],foreg = fchoice[i])
                rec_img = ((thres_img[:,:,i])[:,:,np.newaxis]*rec_img).astype(np.uint8)
                # rec_img[:,:,i] = ((thres_img[:,:,i])*rec_img[:,:,i]).astype(np.uint8)
                self.viewimg(rec_img)
                self.viewimg(thres_img[:,:,i])
            return(thres_img[:,:,i])

# Funtion to extract contours
    def img_contour(self, img):
        #----------------------------------------------------------
        # Contour extraction function
        # Inputs: img - The binary image
        # Outputs: Binary image with contour
        # Searches clockwise from any white pixel
        #-----------------------------------------------------------
        bpix = np.array([[-1,-1]])  # Array to store boundary pixels
        pimg = np.zeros((img.shape[0]+2,img.shape[1]+2))
        pimg[1:-1,1:-1] = img
        clk_ind = np.array([[-1,-1],[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1],[0,-1]])
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                bpixq = False   # Indicator for boundary pixel
                # Check if it is a boundary pixel
                if img[i,j] == 1:
                    for m in range(8):
                        # If one of the neighbours is black then boundary pixel
                        if pimg[i+1+clk_ind[m,0],j+1+clk_ind[m,0]] == 0:
                            bpixq = True
                            break
                # 3 Checks, 1: if white pixel, 2: if boundary pixel,
                #   3: if already in one of the boundary pixels
                if(bpixq and sum(np.all((bpix-[i,j])==0, 1))==0):
                    i1, j1 = i, j
                    pk = 0  # position of previous bpix
                    while True:
                        for m in range(8):
                            k = (m+pk+1)%8
                            # Check if pixel is white, if so step to it
                            if pimg[i1+1+clk_ind[k,0],j1+1+clk_ind[k,1]] == 1:
                                # add it to boundary pixels
                                bpix = np.vstack((bpix, [i1+clk_ind[k,0],j1+clk_ind[k,1]]))
                                # Update position of current bpix
                                i1 = i1+clk_ind[k,0]
                                j1 = j1+clk_ind[k,1]
                                pk = (k+4)%8    # Start index for next pixel
                                break
                        # On returning to start pixel break
                        if i1 == i and j1 == j:
                            break
        bpix = bpix[1:,:]   # Remove first row of [-1,-1]
        # Compose result image
        res_img1 = np.zeros(img.shape)
        res_img1[bpix[:,0],bpix[:,1]]=1 # set boundary pixels to 1

        # Method 2: Dilates the image and subtracts the eroded image
        ker = np.ones((2,2),np.uint8)
        res_img2 = cv2.dilate(img,ker,iterations=1) - cv2.erode(img,ker,iterations=1)
        return(res_img1, res_img2)

# Kernel Variance function
    def ker_var(self,img, N):
        #----------------------------------------------------------
        # Computes variance for texture segmentation
        # Inputs: img, Kernel size
        # Outputs: array with variance at each pixel
        #-----------------------------------------------------------
        pimg = np.zeros((img.shape[0] + N-1,img.shape[1] + N-1))    # Padded image
        pimg[int(N/2):-int(N/2),int(N/2):-int(N/2)] = img
        var_img = np.zeros(img.shape)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                var_img[i,j] = pimg[i:i+N,j:j+N].var() #var = mean(abs(x - x.mean())**2)
        #Scale to 255
        var_img = (5000*var_img/var_img.max())   #var is 0-max
        var_img[np.where(var_img>255)] =255     # Scale appropriately
        return(var_img)

# Texutre based segementation function
    def tex_seg(self,img):
        #----------------------------------------------------------
        # Texture segmentation uses 3 kernel sizes 3,5,7 and uses variance
        #   of gray level histograms to generate 3 channels
        # Then Otsu is applied on this to obtain the resulting image
        # Inputs: img - Gray image
        # Outputs: Three channel variance image
        #-----------------------------------------------------------
        t_img = np.zeros((img.shape[0],img.shape[1],3))
        for i in range(3):
            t_img[:,:,i] = self.ker_var(img,2*i+3)
        return(t_img)

# Function to view image
    def viewimg(self, img):
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
        # Initialize images
        img_rgb = cv2.imread(self.img_pth[img_id])  # RGB
        img_gry = cv2.cvtColor(img_rgb,cv2.COLOR_BGR2GRAY)  # GRAY
        img_hsv = cv2.cvtColor(img_rgb,cv2.COLOR_BGR2HSV)   # HSV
        img_txt = self.tex_seg(img_gry) # Variance texture image
        cv2.imwrite(('img' + str(img_id) +'_txt1.jpg'),(img_txt[:,:,0]).astype(np.uint8))
        cv2.imwrite(('img' + str(img_id) +'_txt2.jpg'),(img_txt[:,:,1]).astype(np.uint8))
        cv2.imwrite(('img' + str(img_id) +'_txt3.jpg'),(img_txt[:,:,2]).astype(np.uint8))
        # Color threshodling for segmentation--------------------------------
        ker = np.ones((20,20),np.uint8) # kernel for filling holes
        # Perform multi otsu for HSV image
        res_img = self.multi_otsu(img_hsv,1,self.ot_choice[img_id][0,:]).astype(np.float)
        res_hsv = res_img #store
        cv2.imwrite(('img' + str(img_id) +'_omask_hsv.jpg'),(255*res_img).astype(np.uint8))
        cv2.imwrite(('img' + str(img_id) +'_hsv.jpg'),(res_img[:,:,np.newaxis]\
        *img_rgb).astype(np.uint8))
        # Dilate and erode image to remove holes
        res_img = cv2.morphologyEx(res_img,cv2.MORPH_CLOSE, ker)
        cv2.imwrite(('img' + str(img_id) +'_ed_hsv.jpg'),(res_img[:,:,np.newaxis]\
        *img_rgb).astype(np.uint8))
        c_img1,c_img2 = self.img_contour(res_img)   # Get contours
        cv2.imwrite(('img' + str(img_id) +'c1_hsv.jpg'),255*c_img1)
        cv2.imwrite(('img' + str(img_id) +'c2_hsv.jpg'),255*c_img2)
        #----------------------------------------------------------------------
        # On BGR images
        # Perform multi otsu for rgb image
        res_img = self.multi_otsu(img_rgb,1,self.ot_choice[img_id][1,:]).astype(np.float)
        res_rgb = res_img #store
        cv2.imwrite(('img' + str(img_id) +'_omask_rgb.jpg'),(255*res_img).astype(np.uint8))
        cv2.imwrite(('img' + str(img_id) +'_rgb.jpg'),(res_img[:,:,np.newaxis]\
        *img_rgb).astype(np.uint8))
        # Dilate and erode image to remove holes
        res_img = cv2.morphologyEx(res_img,cv2.MORPH_CLOSE, ker)
        cv2.imwrite(('img' + str(img_id) +'_ed_rgb.jpg'),(res_img[:,:,np.newaxis]\
        *img_rgb).astype(np.uint8))
        c_img1,c_img2 = self.img_contour(res_img)   # Get contours
        cv2.imwrite(('img' + str(img_id) +'c1_rgb.jpg'),255*c_img1)
        cv2.imwrite(('img' + str(img_id) +'c2_rgb.jpg'),255*c_img2)
        #----------------------------------------------------------------------
        # On Texture images
        # Perform multi otsu for txt image
        res_img = self.multi_otsu(img_txt,1,self.ot_choice[img_id][2,:]).astype(np.float)
        res_txt = res_img
        cv2.imwrite(('img' + str(img_id) +'_omask_txt.jpg'),(255*res_img).astype(np.uint8))
        cv2.imwrite(('img' + str(img_id) +'_txt.jpg'),(res_img[:,:,np.newaxis]\
        *img_rgb).astype(np.uint8))
        # Dilate and erode image to remove holes
        # res_img = cv2.morphologyEx(res_img,cv2.MORPH_CLOSE, ker)
        cv2.imwrite(('img' + str(img_id) +'_ed_txt.jpg'),(res_img[:,:,np.newaxis]\
        *img_rgb).astype(np.uint8))
        c_img1,c_img2 = self.img_contour(res_img)   # Get contours
        cv2.imwrite(('img' + str(img_id) +'c1_txt.jpg'),255*c_img1)
        cv2.imwrite(('img' + str(img_id) +'c2_txt.jpg'),255*c_img2)
        #----------------------------------------------------------------------
        # On Combined masks
        # Or the results
        res_img = np.logical_or(np.logical_or(res_hsv,res_rgb).astype(np.uint8)\
        ,res_txt).astype(np.uint8)
        print(res_img)
        cv2.imwrite(('img' + str(img_id) +'_omask_cmb.jpg'),(255*res_img)\
        .astype(np.uint8))
        cv2.imwrite(('img' + str(img_id) +'_cmb.jpg'),(res_img[:,:,np.newaxis]\
        *img_rgb).astype(np.uint8))
        # Dilate and erode image to remove holes
        res_img = cv2.morphologyEx(res_img,cv2.MORPH_CLOSE, ker)
        cv2.imwrite(('img' + str(img_id) +'_ed_cmb.jpg'),(res_img[:,:,np.newaxis]\
        *img_rgb).astype(np.uint8))
        c_img1,c_img2 = self.img_contour(res_img)   # Get contours
        cv2.imwrite(('img' + str(img_id) +'c1_cmb.jpg'),255*c_img1)
        cv2.imwrite(('img' + str(img_id) +'c2_cmb.jpg'),255*c_img2)


if __name__ == '__main__':
    imgseg = ImageSegment()
    imgseg.main(1)  # imgseg.main(img id)
    imgseg.main(2)
    imgseg.main(3)

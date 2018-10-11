#!/usr/bin/env python3
############################################################################
# Script for homework 4 ECE661
#  Harris corner detector code
#  - Contains functions to find corner points based on Harris corner
#    detection. Then find the correspondence between interest(corner)
#    points from 2 images using SSD (sum of squared differences) or
#    NCC (normalized cross correlation)
# Author: Mythra Balakuntala
############################################################################

import cv2, numpy as np, math, time, matplotlib
import time
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

class HarrisCornerDetector():
    def __init__(self):
        # Image names
        self.img_names = {  1:'HW4Pics/1.jpg',
                            2:'HW4Pics/2.jpg',
                            3:'HW4Pics/rs_truck1.jpg',
                            4:'HW4Pics/rs_truck2.jpg',
                            5:'HW4Pics/rs_book1.jpg',
                            6:'HW4Pics/rs_book2.jpg',
                            7:'HW4Pics/rs_boom1.jpg',
                            8:'HW4Pics/rs_boom2.jpg',
                            }
        # Magic numbers
        self.thr = 2    # Threshold used in r/(1+r)^2 cutoff
        self.lmax_ksize = 31     # For finding local maxima
        self.corr_ksize = 21     # For finding correspondece b/w cornr pts
        self.ncc_thres = 0.75
        self.ssd_thres = 0.15
    # Convolution functions
    def conv_filt(self, filt, img):
        #----------------------------------------------------------
        # Function to convolve the filter across the image
        # Inputs: filt - filter matrix, img -source imgae
        # Outputs: fimg -  filtered image
        #-----------------------------------------------------------
        m = filt.shape[0]
        filt = np.reshape(filt,[1,-1])
        fimg = np.zeros(img.shape)
        # Pad images for convolving
        img = np.hstack([np.zeros((img.shape[0],m)),img,np.zeros((img.shape[0],m))])
        img = np.vstack([np.zeros((m,img.shape[1])),img,np.zeros((m,img.shape[1]))])
        # Apply filters at each point
        for r in range(fimg.shape[0]):
            for c in range(fimg.shape[1]):
                fimg[r,c] = filt@np.reshape(img[int(r+m/2):int(r+3*m/2),int(c+m/2):int(c+3*m/2)],(-1,1))
        return(fimg)

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

# Problem specific codes.....

    # Haar filtered image
    def haar_img(self, img, sigma):
        #----------------------------------------------------------
        # Function to compute haar filtered images along x and y
        # Inputs: img - source image path, sigma - kernel
        # Outputs: imgdx, imgdy
                  # imgdx - Haar filtered along x (applied dx across a row)
                  # imgdy - Haar filtered along y (applied dy down a column)
        #-----------------------------------------------------------
        # Get gaussian smoothed image at the given scale
        img = cv2.GaussianBlur(img,(5,5),sigma)
        m = int(np.ceil(4*sigma) + np.ceil(4*sigma)%2)       # Size of kernel
        dxker = np.ones((m,m))  # xkernel
        dxker[:,int(m/2):] = -1
        dximg = self.conv_filt(dxker, img)      # derivative along x
        dyimg = self.conv_filt(dxker.T, img)    # derivative along y
        # Apply Haar filters at each point
        # return(img)
        return(dximg, dyimg)

    # The corner pts function
    def cpt_mat(self, dximg, dyimg, sigma, tscale=1):
        #----------------------------------------------------------
        # Function to compute corner points using Harris method
        # Inputs: dximg - x derivative matrix, dyimg - y derivative matrix
        #       sigma - scale, tscale - threshold scaling, i.e. only pts with
        #       r/(1+r^2) > mean(r) + tscale*std(r) are chosen as corner pts
        # Outputs: cornerpts - boolean matrix with true only at cornerpts
        #-----------------------------------------------------------
        # We sum in a 5 simga x 5 simga neighbourhood
        m = int(np.ceil(5*sigma) + (np.ceil(5*sigma)+1)%2)
        filt = np.ones((m,m))
        # Find the three components of C matrix
        c = np.zeros((dximg.shape[0],dximg.shape[1],3))
        c[:,:,0] = self.conv_filt(filt, dximg*dximg)  # dx^2 components
        c[:,:,1] = self.conv_filt(filt, dximg*dyimg)  # dx*dy components
        c[:,:,2] = self.conv_filt(filt, dyimg*dyimg)  # dy^2 components
        detc = c[:,:,0]*c[:,:,2] - c[:,:,1]*c[:,:,1]  # determinant of C
        trc = (c[:,:,0] + c[:,:,2])                  # trace of C
        mod_r = detc/(trc*trc)   # r/(1+r^2)
        # ^ There will be some divisions of 0/0 resulting in nan and warning...
        mod_r = np.nan_to_num(mod_r)    # Eliminate nan
        # Find the corner points with threshold
        thres = mod_r.mean() + tscale*mod_r.std()     # Take threshold
        # Create kernel to find local maxima, we use a 31x31 kernel
        ker = np.ones((self.lmax_ksize,self.lmax_ksize))
        temp = cv2.dilate(mod_r, ker)
        temp = cv2.compare(mod_r,temp,cv2.CMP_EQ)
        return(temp>thres)

    # Find correspondences
    def corr_pts(self, img1, img2, cpts1, cpts2, mtype=1):
        #----------------------------------------------------------
        # Function to compute correspondence b/w obtained corner points
        # Inputs: img1 - first image, img2 - second image,
        #   cpts1 - 1st img corner pts, cpts2 - 2nd img corner pts
        #   thres - threshold for cutoff, mtype - Method used (0-SSD, 1-NCC)
        # Outputs: corr_mat - Array of correspondence pairs
        #-----------------------------------------------------------
        cpts_id1 =  np.where(cpts1==True)# indices of corner points 1
        cpts_id2 =  np.where(cpts2==True)# indices of corner points 2
        m = int(self.corr_ksize/2)
        # Pad image 1 for convolving
        pimg1 = np.hstack([np.zeros((img1.shape[0],m)),img1,np.zeros((img1.shape[0],m))])
        pimg1 = np.vstack([np.zeros((m,pimg1.shape[1])),pimg1,np.zeros((m,pimg1.shape[1]))])
        # Pad image 2 for convolving
        pimg2 = np.hstack([np.zeros((img2.shape[0],m)),img2,np.zeros((img2.shape[0],m))])
        pimg2 = np.vstack([np.zeros((m,pimg2.shape[1])),pimg2,np.zeros((m,pimg2.shape[1]))])
        #
        ssd = np.zeros(1)
        ind = np.array([0,0,0,0])
        # Compute the ssd/ncc for each point
        for i in range(cpts_id1[0].shape[0]):
            pssd = 10**5
            idx = 0
            for j in range(cpts_id2[0].shape[0]):
                # Get the corresponding value matrix for each matrix
                f1 = pimg1[cpts_id1[0][i]:cpts_id1[0][i]+2*m \
                ,cpts_id1[1][i]:cpts_id1[1][i]+2*m]
                f2 = pimg2[cpts_id2[0][j]:cpts_id2[0][j]+2*m \
                ,cpts_id2[1][j]:cpts_id2[1][j]+2*m]
                # Check if method is ssd or ncc
                if mtype == 0 :  # ssd
                    f_ssd = sum(sum((f2-f1)**2))    # Find ssd
                    if f_ssd< pssd:
                        pssd = f_ssd
                        idx = j
                elif mtype ==1: # ncc
                    # Compute NCC
                    f_ncc = sum(sum((f1 - f1.mean())*(f2 - f2.mean())))
                    f_ncc = f_ncc/np.sqrt(sum(sum((f1 - f1.mean())**2))*\
                            sum(sum((f2 - f2.mean())**2)))
                    if abs(f_ncc) > self.ncc_thres:
                        # Update index array if greater than thres
                        ind = np.vstack((ind,np.array([cpts_id1[0][i],\
                        cpts_id1[1][i],cpts_id2[0][j],cpts_id2[1][j]])))
            if mtype == 0:
                # Update index array if greater than thres
                ind = np.vstack((ind,np.array([cpts_id1[0][i],\
                cpts_id1[1][i],cpts_id2[0][idx],cpts_id2[1][idx]])))
                ssd = np.append(ssd,pssd)
        # IF ssd threshold based on max and cutoff
        if mtype ==0:
            ind = ind[np.where(ssd<self.ssd_thres*ssd[1:].mean())[0],:]
        print(ind.shape)
            #ind = ind[:100,:] # Choose the first n numbers
        return([ind[1:,0],ind[1:,1],ind[1:,2],ind[1:,3]])

    # Function to draw draw matches
    def draw_matches(self, img1, img2, kp1, kp2):
        r1, c1 = img1.shape[0:2]
        r2, c2 = img2.shape[0:2]
        # output image
        res_img = np.zeros((max([r1, r2]), c1 + c2, 3))
        # Add first image
        res_img[0:r1, 0:c1, :] = np.dstack((img1,img1, img1))
        res_img[:r2, c1:, :] = np.dstack([img2,img2, img2]) # Add second image
        # Line and ponit properties
        rad = 4 # For point
        RED = (0,0,255)
        BLUE = (255,0,0)
        GREEN = (0,255,0)
        thickness = 2
        # Draw partition line
        cv2.line(res_img, (c1,0),(c1,max([r1,r2])),(0,0,0),thickness)
        # Get matches and draw lines
        for i in range(kp1.shape[0]):
            # Check angle
            th = math.atan2(kp2[i,0]-kp1[i,0],kp2[i,1]+c1-kp1[i,1])*180/math.pi
            # Filter points based on angle
            if abs(th) < 5: #Check if angle is less than desired
                # Draw circle for points
                cv2.circle(res_img,(kp1[i,1],kp1[i,0]),rad,GREEN,thickness) # img1 pt
                cv2.circle(res_img,(kp2[i,1]+c1,kp2[i,0]),rad,GREEN,thickness) # img2 pt
                # Draw line between the points
                cv2.line(res_img, (kp1[i,1],kp1[i,0]),(kp2[i,1]+c1,kp2[i,0]),BLUE,thickness)
            # Uncomment to show false correspndences
            # else:
            #     # Probably false correspondences
            #     cv2.line(res_img, (kp1[i,1],kp1[i,0]),(kp2[i,1]+c1,kp2[i,0]),RED,1)
        return(res_img.astype(np.uint8))

    # The main function
    def main(self, img1_id, img2_id, sigma):
        #----------------------------------------------------------
        # The main function
        # Inputs: img1_id - Id for the first image,
        #   img2_id - Id for the 2nd image, sigma
        # Outputs: corr_mat - Array of correspondence pairs
        #-----------------------------------------------------------
        # Find corner points for image 1
        img1_rgb = cv2.imread(self.img_names[img1_id])
        img1 = cv2.cvtColor(img1_rgb,cv2.COLOR_BGR2GRAY)
        dximg1, dyimg1  = self.haar_img(img1,sigma)
        cpts1 = self.cpt_mat(dximg1, dyimg1, sigma,self.thr)
        # Find corner points for image 2
        img2_rgb = cv2.imread(self.img_names[img2_id])
        img2 = cv2.cvtColor(img2_rgb,cv2.COLOR_BGR2GRAY)
        dximg2, dyimg2  = self.haar_img(img2,sigma)
        cpts2 = self.cpt_mat(dximg2, dyimg2, sigma,self.thr)
        # Find corresponding corner pts using NCC
        corr_cpts = self.corr_pts(img1, img2, cpts1, cpts2, 1)
        kp1 = np.array([corr_cpts[0],corr_cpts[1]]).T
        kp2 = np.array([corr_cpts[2],corr_cpts[3]]).T
        img = self.draw_matches(img1, img2, kp1, kp2)
        cv2.imwrite('img'+str(img1_id)+'ncc_s'+str(int(sigma)) +'.jpg',img)
        # Find corresponding corner pts using SSD
        corr_cpts = self.corr_pts(img1, img2, cpts1, cpts2, 0)
        kp1 = np.array([corr_cpts[0],corr_cpts[1]]).T
        kp2 = np.array([corr_cpts[2],corr_cpts[3]]).T
        img = self.draw_matches(img1, img2, kp1, kp2)
        cv2.imwrite('img'+str(img1_id)+'ssd_s'+str(int(sigma))+'.jpg',img)

if __name__=="__main__":
    hcd = HarrisCornerDetector()
    for i in range(1,5):
        for j in range(1,5):
            print(i,j)
            hcd.main(2*i-1, 2*i, 1.4*j)    # Image pair 1 with scale 1.2

#!/usr/bin/env python3
############################################################################
# Script for homework 4 ECE661
# Surf interest point detection and matching
# Author: Mythra Balakuntala
############################################################################

import cv2, numpy as np, math, time, matplotlib, random
import time
from hw3 import imageTr

class AutoHomographyMatch():
    def __init__(self):
        # Image names
        self.img_names = {  1:'HW5Pics/1.jpg',
                            2:'HW5Pics/2.jpg',
                            3:'HW5Pics/3.jpg',
                            4:'HW5Pics/4.jpg',
                            5:'HW5Pics/5.jpg',
                            0:'HW5Pics/0.jpg',
                            }
        self.imgTr = imageTr()

    # Facotorial function
    def fact(self, n):
        if n == 0:
            return(1)
        else:
            return(n*self.fact(n-1))

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

# Function to draw draw matches
    def draw_matches(self, img1, img2, kp1, kp2, inliers):
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
            if i in inliers[0]:
                # Draw circle for points
                cv2.circle(res_img,(kp1[i,1],kp1[i,0]),rad,GREEN,thickness) # img1 pt
                cv2.circle(res_img,(kp2[i,1]+c1,kp2[i,0]),rad,GREEN,thickness) # img2 pt
                # Draw line between the points
                cv2.line(res_img, (kp1[i,1],kp1[i,0]),(kp2[i,1]+c1,kp2[i,0]),BLUE,thickness)
            # else:
            #     # Draw circle for points
            #     cv2.circle(res_img,(kp1[i,1],kp1[i,0]),rad,RED,thickness) # img1 pt
            #     cv2.circle(res_img,(kp2[i,1]+c1,kp2[i,0]),rad,RED,thickness) # img2 pt
            #     # Draw line between the points
            #     cv2.line(res_img, (kp1[i,1],kp1[i,0]),(kp2[i,1]+c1,kp2[i,0]),RED,thickness)
        return(res_img.astype(np.uint8))

    # The feature match function
    def feature_match(self, img1, img2, thres = 0.8):
        #----------------------------------------------------------
        # The freature matching function
        # Uses Surf for getting features
        # Inputs: img1 - Image 1, img2 - Image 2, thres - threshold for match
        # Outputs: kpts1, kpts2 - pixel coordinates of matches
        #        kpts1[i] matches with kpts2[i]
        #-----------------------------------------------------------
        hthres = 1000   # Hessian threshold
        # Create surf handle
        surf = cv2.xfeatures2d.SURF_create(hthres, extended = True)
        # Find corner points for image 1
        img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        kp1, dcpt1 = surf.detectAndCompute(img1, None)  # Keypoints and descriptors img1
        # Image 2 features
        img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        kp2, dcpt2 = surf.detectAndCompute(img2, None)
        # Use inbuilt cv2 functions to draw
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(dcpt1, dcpt2, k=2)
        good = []
        for m,n in matches:
            if m.distance < thres*n.distance:
                good.append([m])
        # Get matching pixel coordinates
        kpts1 = np.ones((len(good),3))
        kpts2 = np.ones((len(good),3))
        for i in range(len(good)):
            kpts1[i,0:2] = np.round(kp1[good[i][0].queryIdx].pt)
            kpts2[i,0:2] = np.round(kp2[good[i][0].trainIdx].pt)
        kpts1[:,[0, 1]] = kpts1[:,[1, 0]]
        kpts2[:,[0, 1]] = kpts2[:,[1, 0]]
        kpts1 = kpts1.astype(int)
        kpts2 = kpts2.astype(int)
        out = np.zeros(img1.shape)
        # img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,out,(255,0,0),(0,255,0),flags=2)
        img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,out,flags=2)
        # self.viewimg(img)
        self.viewimg(img3)
        return(kpts1, kpts2)

        # RANSAC to reject outliers
    def ransac(self, kpts1, kpts2):
        #----------------------------------------------------------
        # The Random sample and consensus function
        # Randomly chooses points and eavaluates to find the outliers
        # Inputs: kpts1, kpts2 - pixel coordinates of matches
        #        kpts1[i] matches with kpts2[i]
        # Outputs: inpairs - inlier pairs
        #-----------------------------------------------------------
        # eps is assumed to be 0.3, i.e. 30% of all correspondences is false
        # This gives number of trails to perform as 26 (using 5 pairs for H)
        #   i.e. eps = 0.2, n = 5, N = ln(1-p)/ln(1-(1-eps)^n) = 25.03
        npts = kpts1.shape[0]
        self.npairs = 5
        self.N = min(37, self.fact(npts)/(self.fact(npts-self.npairs)*self.fact(self.npairs))) # Number of trials
        self.delta = 10   # Threshold for position error
        #__________________________________________________________________
        perr = 10000    # Random large prev error
        h_min = np.zeros((3,3))
        insize = 0  # inlier set size
        # print(np.hstack((kpts1, kpts2)))
        # print(kpts1[np.where(kpts1[:,1]>800),:])
        # print('*'*50)
        for i in range(self.N):
            # Choose n random pairs
            j = random.sample(range(kpts1.shape[0]),self.npairs)
            h = self.imgTr.computeHSvd(kpts1[j,:],kpts2[j,:])
            prj_pts = (h@kpts1.T).T
            prj_pts = prj_pts/prj_pts[:,-1][:,np.newaxis]
            prj_pts = prj_pts.astype(int)
            err = np.linalg.norm(prj_pts - kpts2,axis=1)[:,None].astype(int)
            in_ind = np.where(err<self.delta)
            if in_ind[0].shape[0] == 0:
                continue
            else:
                err = err[in_ind]
            print(err.shape[0])
            if insize < err.shape[0]:   # If greater inlier set
                inset = in_ind
                h_min = h
        # h_min = self.imgTr.computeHSvd(kpts1[inliers[0],:],kpts2[inliers[0],:])
        print(inset)
        prj_pts = (h_min@kpts1[inset[0],:].T).T
        prj_pts = prj_pts/prj_pts[:,-1][:,np.newaxis]
        prj_pts = prj_pts.astype(int)
        # print(np.hstack((prj_pts, kpts2[inliers[0],:], np.linalg.norm(prj_pts - kpts2[inliers[0],:],axis=1)[:,None].astype(int))))
        return(h_min, inset)

        # The main function
    def main(self, n, ref_img_id):
        #----------------------------------------------------------
        # The main function
        # Loads all images and calls other functions to compute homography
        #   and project all images onto given image.
        # Inputs: n - Compute for 'n' images in self.img_names
        #   ref_img_id - The id for image to be projected onto
        # Writes output image in the folder
        #-----------------------------------------------------------
        h = np.zeros((9,n-1))   # The homography matrices (each col for 1 pair of imgs)
        for i in range(n-1):
            img1_rgb = cv2.imread(self.img_names[i+1])
            img2_rgb = cv2.imread(self.img_names[i+2])
            img1 = cv2.cvtColor(img1_rgb,cv2.COLOR_BGR2GRAY)
            img2 = cv2.cvtColor(img2_rgb,cv2.COLOR_BGR2GRAY)
            # Get matching pixels
            kpts1, kpts2 = self.feature_match(img1_rgb, img2_rgb)
            # print(np.hstack((kpts1, kpts2)))
            # print(kpts1[np.where(kpts1[:,1]>800),:])
            # print('*'*50)
            # Reject outliers using RANSAC
            h_min, inliers = self.ransac(kpts1, kpts2)
            h_min = np.linalg.pinv(h_min)
            h_min = h_min/h_min[-1,-1]
            h_cv, s = cv2.findHomography(kpts1, kpts2, cv2.RANSAC, 1)
            print(h_cv)
            print(np.linalg.pinv(h_cv)/np.linalg.pinv(h_cv)[-1,-1])
            img = self.draw_matches(img1, img2, kpts1, kpts2, inliers)
            h[:,i][:,None] = h_min.reshape((-1,1))
            # Find boudary of dest_image
            # ht,wt,crnr_pts, h_new,src_pts = self.imgTr.cornerpts(self.img_names[i+1], h_min)
            # Create empty image
            # dest_img = np.zeros(img1_rgb.shape[0:2])
            # cv2.imwrite(self.img_names[0],dest_img)
            # crnr_pts = np.array([[0,0,1],[0,dest_img.shape[1],1],[dest_img.shape[0],dest_img.shape[1],1],[dest_img.shape[0],0,1]])
            # Get projected image
            # dest_img = self.imgTr.getProjImg(src_pts, crnr_pts , self.img_names[i+1],self.img_names[0],np.linalg.pinv(h_new))
            dest_img1 = cv2.warpPerspective(cv2.cvtColor(img1_rgb,cv2.COLOR_BGR2GRAY),h_min,img1_rgb.shape[0:2])
            dest_img2 = cv2.warpPerspective(cv2.cvtColor(img1_rgb,cv2.COLOR_BGR2GRAY),h_cv,img1_rgb.shape[0:2])
            self.viewimg(dest_img1)
            self.viewimg(dest_img2)
            # cv2.imwrite('p1_modimg'+str(img_ind)+'.jpg',dest_img)

        # cv2.imwrite('img'+str(img1_id)+'surf_s'+str(int(sigma))+'.jpg',img3)

if __name__=="__main__":
    auhoma = AutoHomographyMatch()
    auhoma.main(2,2)

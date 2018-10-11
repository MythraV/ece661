#!/usr/bin/env python3
############################################################################
# Script for homework 4 ECE661
# Surf interest point detection and matching
# Author: Mythra Balakuntala
############################################################################

import cv2, numpy as np, math, time, matplotlib
import time

class SurfCornerDetector():
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
        self.ed_thres = 20

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
            # Draw circle for points
            cv2.circle(res_img,(kp1[i,1],kp1[i,0]),rad,GREEN,thickness) # img1 pt
            cv2.circle(res_img,(kp2[i,1]+c1,kp2[i,0]),rad,GREEN,thickness) # img2 pt
            # Check angle
            th = math.atan2(kp2[i,0]-kp1[i,0],kp2[i,1]+c1-kp1[i,1])*180/math.pi
            if abs(th) > 5:
                # Draw line between the points
                cv2.line(res_img, (kp1[i,1],kp1[i,0]),(kp2[i,1]+c1,kp2[i,0]),RED,thickness)
            else:
                # Draw line between the points
                cv2.line(res_img, (kp1[i,1],kp1[i,0]),(kp2[i,1]+c1,kp2[i,0]),BLUE,thickness)
        return(res_img.astype(np.uint8))

    # Find correspondences
    def corr_pts(self, img1, img2, kp1, kp2, cpts1, cpts2):
        #----------------------------------------------------------
        # Function to compute correspondence b/w obtained corner points
        # Inputs: img1 - first image, img2 - second image,
        #   cpts1 - 1st img corner pts, cpts2 - 2nd img corner pts
        #   thres - threshold for cutoff
        # Outputs: corr_mat - Array of correspondence pairs
        #-----------------------------------------------------------
        ed = np.zeros(1)
        ind = np.array([0,0,0,0])
        # Compute the ssd/ncc for each point
        for i in range(cpts1.shape[0]):
            ped = 10**5
            idx = 0
            for j in range(cpts2.shape[0]):
                # Get the corresponding value matrix for each matrix
                f_ed = np.sqrt(sum((cpts1[i]-cpts2[j])**2))    # Euclidean dist
                if f_ed< ped :
                    ped = f_ed
                    idx = j
            # Update index array if greater than thres
            ind = np.vstack((ind,np.array([kp1[i].pt[0],\
            kp1[i].pt[1],kp2[idx].pt[0],kp2[idx].pt[1]])))
            ed = np.append(ed,ped)
            if i%100 == 0:
                print(i)
        # IF ed threshold based on max and cutoff
        ind = ind[np.where(ed<self.ed_thres*ed[1:].min())[0],:].astype(int)
        return([ind[1:,0],ind[1:,1],ind[1:,2],ind[1:,3]])

    # The main function
    def main(self, img1_id, img2_id, sigma, hthres, thr, draw_flag=True):
        #----------------------------------------------------------
        # The main function
        # Inputs: img1_id - Id for the first image,
        #   img2_id - Id for the 2nd image, sigma
        # Outputs: corr_mat - Array of correspondence pairs
        #-----------------------------------------------------------
        self.ed_thres = thr
        # Surf
        surf = cv2.xfeatures2d.SURF_create(hthres, extended = True)
        # Find corner points for image 1
        img1_rgb = cv2.imread(self.img_names[img1_id])
        img1 = cv2.cvtColor(img1_rgb,cv2.COLOR_BGR2GRAY)
        kp1, dcpt1 = surf.detectAndCompute(img1, None)  # Keypoints and descriptors img1
        surf = cv2.xfeatures2d.SURF_create(hthres, extended = True)
        # Image 2 features
        img2_rgb = cv2.imread(self.img_names[img2_id])
        img2 = cv2.cvtColor(img2_rgb,cv2.COLOR_BGR2GRAY)
        kp2, dcpt2 = surf.detectAndCompute(img2, None)
        # Use inbuilt cv2 functions to draw
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(dcpt1, dcpt2, k=2)
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
        out = np.zeros(img1.shape)
        img3 = cv2.drawMatchesKnn(img1_rgb,kp1,img2_rgb,kp2,good,out,(255,0,0),(0,255,0),flags=2)

        # Use written functions to draw
        corr_cpts = self.corr_pts(img1, img2, kp1, kp2, dcpt1, dcpt2)
        kp1 = np.array([corr_cpts[0],corr_cpts[1]]).T
        kp2 = np.array([corr_cpts[2],corr_cpts[3]]).T
        # print(corr_cpts)
        img = self.draw_matches(img1, img2, kp1, kp2)

        if draw_flag == True:
            cv2.imwrite('img'+str(img1_id)+'surf_s'+str(int(sigma))+'.jpg',img3)
        # If using own written draw function instead of one from cv2
        # else:
        #     cv2.imwrite('img'+str(img1_id)+'surf_s'+str(int(sigma))+'.jpg',img)

if __name__=="__main__":
    sfd = SurfCornerDetector()
    #sfd.main(1, 2, 1.4)    # Image pair 1 with scale 1.2
    for i in range(1,5):
        sfd.main(2*i-1, 2*i, 1.4, 4000, 8)    # Image pair 1 with scale 1.2

#!/usr/bin/env python3
############################################################################
# Script for homework 8 ECE661
# Camera callibration file
#  - Contains functions to detect corner points of chekered board pattern
#       Uses opencv canny and hough to find corners and LM to refine them.
#  - Functions to perform camera callibration using Zhang's algorithm
#       Uses corner poitns to estimate intrinsic, then extrinsic and
#       finally LM to refine parameters
# Outputs : Camera intrinsic paramters, extrinsic paramters (w.r.t specified
#   image), distortion paramters.
# Author: Mythra Balakuntala
############################################################################
import numpy as np, cv2
from scipy.optimize import least_squares as ls
from scipy import linalg as slg
import matplotlib.pyplot as plt
# from scipy.optimize import leastsq as lsq
class ImageReconstruct():

    def __init__(self):
        self.kill = False
        self.c = [list(),list()]
        self.img = [np.array([]), np.array([])]
# Function to view image
    def viewimg(self, img, scale = 600):
        if not self.kill:
            cv2.namedWindow('resultImage',cv2.WINDOW_NORMAL)
            cv2.imshow('resultImage',img)
            cv2.resizeWindow('resultImage', scale, scale)
            key = 1
            print('Press q to quit')
            while key !=ord('q'):       # Quit on pressing q
                key = cv2.waitKey(0)
                if key == ord('s'):
                    self.kill = True
            cv2.destroyAllWindows()

    def mouse_cb(self, event, x, y, flags, img_id):
        if event ==cv2.EVENT_LBUTTONDOWN:
            self.c[img_id].append([x,y])
            cv2.circle(self.img[img_id], (x,y), 1, (255,0,0),2)
            cv2.imshow('Image',self.img[img_id])

# Function to draw draw matches
    def draw_matches(self, img1, img2, kp1, kp2):
        r1, c1 = img1.shape[0:2]
        r2, c2 = img2.shape[0:2]
        # output image
        res_img = np.zeros((max([r1, r2]), c1 + c2, 3))
        # Add first image
        res_img[0:r1, 0:c1, :] = img1
        res_img[:r2, c1:, :] = img2 # Add second image
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
            # if i in inliers[0]:
            # Draw circle for points
            cv2.circle(res_img,(kp1[i,0],kp1[i,1]),rad,GREEN,thickness) # img1 pt
            cv2.circle(res_img,(kp2[i,0]+c1,kp2[i,1]),rad,GREEN,thickness) # img2 pt
            # Draw line between the points
            cv2.line(res_img, (kp1[i,0],kp1[i,1]),(kp2[i,0]+c1,kp2[i,1]),BLUE,thickness)
            # else:
            #     # Draw circle for points
            #     cv2.circle(res_img,(kp1[i,1],kp1[i,0]),rad,RED,thickness) # img1 pt
            #     cv2.circle(res_img,(kp2[i,1]+c1,kp2[i,0]),rad,RED,thickness) # img2 pt
            #     # Draw line between the points
            #     cv2.line(res_img, (kp1[i,1],kp1[i,0]),(kp2[i,1]+c1,kp2[i,0]),RED,thickness)
        return(res_img.astype(np.uint8))

#this function will be called whenever the mouse is right-clicked
    def get_points(self, img_id):
    #right-click event value is 2
        cv2.namedWindow('Image',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Image', 1000, 1000)
        cv2.setMouseCallback('Image', self.mouse_cb, img_id)
        cv2.imshow('Image',self.img[img_id])
        key = 1
        while key !=ord('q'):       # Quit on pressing q
            key = cv2.waitKey(0)
        cv2.destroyAllWindows()
# Skew symmetric matrix from vector
    def vskew(self, e):
        return(np.array([[0,-e[2],e[1]],[e[2],0,-e[0]],[-e[1],e[0],0]]))

# function to normalize x
    def normalize(self, x):
        # Mean
        m = np.mean(x,0)
        d = x - m  # distance to mean
        mdis = np.mean(np.sum(d**2,1))
        s = np.sqrt(2)/mdis
        tmat = np.array([[s ,0, -s*m[0]],[0,s,-s*m[1]],[0,0,1]])
        e = np.ones((x.shape[0],1))
        y = np.hstack((x,e))
        return((tmat@(y.T)).T, tmat)

    def find_fundemental(self, x1, x2, t1, t2):
        n = x1.shape[0]
        # Compute the A mat
        A = np.zeros((n,9))
        for i in range(n):
            A[i,:] = [x2[i,0]*x1[i,0], x2[i,0]*x1[i,1], x2[i,0]\
            ,x2[i,1]*x1[i,0], x2[i,1]*x1[i,1], x2[i,1]\
            ,x1[i,0], x1[i,1], 1]
        u,s,vh = np.linalg.svd(A)
        f = (vh.T)[:,-1]
        fmat = f.reshape(3,3)
#--- Debug
        # for i in range(n):
        #     print(x2[i,:].T@fmat@x1[i,:])
#---
        # Reduce rank of F to 2
        u,s,vh = np.linalg.svd(fmat)
        s[-1] = 0   # set last singular value to 0
        fmat = u@np.diag(s)@vh
        # Denormalize
        fmat = t2.T@fmat@t1
        # Set last term to 1
        fmat = fmat/fmat[-1,-1]
        print('Rank of F is ',np.linalg.matrix_rank(fmat))
        # Find epipoles
        # Left epipose ,aka right null space
        e1 = (slg.null_space(fmat).T)[0]
        e2 = (slg.null_space(fmat.T).T)[0]
        return(fmat, e1/e1[-1], e2/e2[-1])

# Compute projection matrices from F and e'
    def get_pmats(self, f, e):
        # Compute projection matrices
        p1 = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1]])
        ex = self.vskew(e)
        p2 = np.hstack((ex@f, e[:,None]))
        return(p1,p2)

# Function to perform reconstruction
    def recon(self, x1, x2 , p1, p2):
        # Compute X from x1,x2 and p1,p2 using traingulation
        n = x1.shape[0]
        # Compute the A mat
        X = np.zeros((n,4))
        for i in range(n):
            A = np.zeros((4,4))
            # From A matrix
            A[0,:] = x1[i,0]*p1[2,:] - p1[0,:]
            A[1,:] = x1[i,1]*p1[2,:] - p1[1,:]
            A[2,:] = x2[i,0]*p2[2,:] - p2[0,:]
            A[3,:] = x2[i,1]*p2[2,:] - p2[1,:]
            # Find X
            u,s,vh = np.linalg.svd(A)
            X[i,:] = (vh.T)[:,-1]
            X[i,:] = X[i,:]/X[i,-1] #Normalize
        return(X)

# Main function
    def main(self):
        self.img[0] = cv2.imread('1.jpg')
        self.img[1] = cv2.imread('2.jpg')
        # Get corresponding points manually,
        # self.get_points(0)    # array of points on left image
        # print(self.c[0])
        # cv2.imwrite('Pic_pts1.jpg',self.img[0])
        # self.get_points(1)    # array of points on right image
        # print(self.c[1])
        # cv2.imwrite('Pic_pts2.jpg',self.img[1])
        self.c[0] = [[328, 285], [395, 207], [248, 172], [148, 227], [167, 331], [323, 399], [384, 307], [340, 257]]
        self.c[1] = [[284, 267], [390, 204], [268, 164], [156, 201], [173, 300], [284, 381], [382, 305], [307, 243]]
        # Draw matches
        kp1 = np.array(self.c[0])
        kp2 = np.array(self.c[1])
        img = self.draw_matches(self.img[0], self.img[1], kp1, kp2)
        # self.viewimg(img)
        cv2.imwrite('Pic_spts.jpg',img)
        # Reinitialize images
        self.img[0] = cv2.imread('1.jpg')
        self.img[1] = cv2.imread('2.jpg')
        # Get normalized x and x'
        x1n,t1 = self.normalize(kp1)
        x2n,t2 = self.normalize(kp2)
        # Get fundemental matrix
        fmat, e1,e2 = self.find_fundemental(x1n,x2n,t1,t2)
        print('The fundamental matrix is ',fmat)
        print('The left epipole is ', e1)
        print('The right epipole is ', e2)
        # Compute projection matrices
        p1, p2 = self.get_pmats(fmat, e2)
        print(p2)
        # Compute X , world points
        X = self.recon(kp1,kp2,p1,p2)
        print(X[:,0])
        # Plot points
        fig = plt.figure()
        ax = fig.axes(projection='3d')
        ax.plot3D(X[:,0],X[:,1],X[:,2])
        plt.show()

if __name__ == '__main__':
    imgrec = ImageReconstruct()
    imgrec.main()

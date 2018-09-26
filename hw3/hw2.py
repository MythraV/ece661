#!/usr/bin/env python3
############################################################################
# Script for homework 2 ECE661
#   Has functions for:
# - finding homographies between sets of coordinates
# - Mapping one image into frame of another image
# -
# Author: Mythra Balakuntala
############################################################################
import cv2, numpy as np, math, time, matplotlib
import time
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


class imageTransform():
    def __init__(self):
        # Arrays containing corresponding points
            # Arrays containing corresponding points
        self.p = { 1 : np.array([[1506,170,1],[2960,722,1],[3002,2046,1],[1490,2240,1]]),
                       2 : np.array([[1326,334,1],[3012,614,1],[3032,1896,1],[1302,2014,1]]),
                       3 : np.array([[926,736,1],[2798,382,1],[2846,2226,1],[900,2088,1]]),
                       11: np.array( [[620,610,1],[870,600,1],[822,1016,1],[568,1006,1]]),
                       12: np.array([[654,600,1],[944,612,1],[928,1010,1],[626,1000,1]]),
                       13: np.array([[946,586,1],[1264,604,1],[1284,1109,1],[950,1126,1]]),
                       10: np.array([[0,0,1],[770,0,1],[770,930,1],[0,930,1]]),
                       0: np.array([[0,0,1],[1200,0,1],[1200,718,1],[0,718,1]])
                       }

        # Create image names dict
        self.img_names = {1:'PicsHw2/1.jpg',
                     2:'PicsHw2/2.jpg',
                     3:'PicsHw2/3.jpg',
                     0:'PicsHw2/Jackie.jpg',
                     11:'PicsHw2/n1.jpg',
                     12:'PicsHw2/n2.jpg',
                     13:'PicsHw2/n3.jpg',
                     10:'PicsHw2/batman.jpg',
                     -1:'PicsHw2/Black.jpg'}

    def getProjImg(self,p0,p1,img0_pth,img1_pth, bnd_type = 1, H = np.zeros(1)):
        # Inputs: p0 - Source image reference points
        #         p1 - Destination image reference points
        #   img0_pth - Path to source image
        #   img1_pth - Path to destination image
        #   bnd_type - Boundary type (1 - Make bounding box around dest and map,
        #                or 0 - Use src image boundary pts to get dest_ ref pts)
        #          H - Homograpy from dest to src, calculated from p0 and p1 if not given

        # computes homography between the frames and maps img1 onto img2
        # Load the images
        img0 = cv2.imread(img0_pth)
        img1 = cv2.imread(img1_pth)
        print(img0.shape, img1.shape)
        # Find homography taking p1 to p0,
        if H.shape[0] == 1:
            H = self.computeHSvd(p1, p0)
        # Limits of bounding box
        # Compute bounding box of distorted image (p1)
        if bnd_type:
            clim = [min(p1[:,0]),max(p1[:,0])]
            rlim = [min(p1[:,1]),max(p1[:,1])]
        elif not bnd_type:
            # Compute from src boundary pts.
            # Compute homography taking src to dest
            Hinv = np.linalg.pinv(H)
            # Find corresponding boundary pts on dest
            pt = (Hinv @ p0[[0,1,-2,-1],:].T).T
            # Normalise to set last elemnt as 1 (x,y,1<)
            pt = (pt.T/pt[:,-1]).T
            # Find limits of bounding box
            clim = np.array([min(pt[:,0]),max(pt[:,0])]).astype(int)
            rlim = np.array([min(pt[:,1]),max(pt[:,1])]).astype(int)

        print("clim is ", clim)
        print("rlim is ", rlim)
        # We iterate over a rectangular bounding box of distorted image PQRS
        # Check if each point is inside desired distorted polygon PQRS
        # If it is we find the correspoinding point on source image and map its
        #   rgb values onto the destination image
        for i in range(clim[0],clim[1]+1):
            for j in range(rlim[0],rlim[1]+1):
                    # Check if point is inside polygon
                    if not self.isinside([i,j],p1[:,0:2]):
                        continue
                    # get point in projected frame
                    projp = H @ np.array([[i],[j],[1]])
                    projp = projp/projp[-1]
                    # Convert to integers
                    projp = projp.astype(int)
                    #img1[projp[1],projp[0],:] = img0[j,i,:]
                    try:
                        img1[j,i,:] = img0[projp[1],projp[0],:]
                    except:
                        continue
        return img1

    def computeHSvd(self, p0, p1):
        # The dimensions of h (h1-h9 elements of H) is 9x1
        # We rewrite Hp0 = p1 as Ah = y
        # Construct A for Ah = y
        nh = p0.shape[1]  # dimensions of H nxn
        A = np.zeros([p0.shape[0]*p0.shape[1],nh**2])   # Based on size of H (9 elements)
        Nx = np.reshape(range(0,nh**2),[nh,nh])
        for i in range(0,len(p0[:,0])):
            for j in range(0,nh**2):
                r_ind = nh*i + np.where(Nx==j)[0][0]
                A[r_ind,j] = p0[i,j%nh]
        # Find solution of Ah = y, using h = (A^T.A)^(-1).A^T.y
        # Construct A' (8x9) from above
        for i in range(0,int(A.shape[0]/3)):
            A[3*i,:] = -A[3*i,:] + p1[i,0]*A[3*i+2,:]
            A[3*i+1,:] = -A[3*i+1,:] + p1[i,1]*A[3*i+2,:]
        r_ind = [x for x in range(A.shape[0]) if (x+1)%3 != 0]
        A = A[r_ind,:]
        # SVD to solve for A
        u, s, v = np.linalg.svd(A)
        null_space = v.T[:,-1]
        h = np.reshape(null_space,(nh,nh))
        h = h/h[nh-1,nh-1]
        return h

    def isinside(self,p, polypts):
        # polypts defines the vertices of the polygon
        # p is the point which is being checked for
        num_pts = len(polypts)
        isin = False
        p1x, p1y = polypts[0]
        for i in range(1, num_pts + 1):
            p2x, p2y = polypts[i % num_pts]
            if p[1] > min(p1y, p2y):
                if p[1] <= max(p1y, p2y):
                    if p[0] <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (p[1] - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or p[0] <= xinters:
                            isin = not isin
            p1x, p1y = p2x, p2y
        return isin

    def main(self, img_num0,img_num):
        print(cv2.__version__)
        # Project image and display
        img = self.getProjImg( self.p[img_num0], self.p[img_num], self.img_names[img_num0],self.img_names[img_num])
        # Show image and destroy on 'q' pressed
        cv2.imwrite('modimg'+str(img_num)+'.jpg',img)
        # self.viewimg(img)


    def viewimg(self,img):
        cv2.namedWindow('resultImage',cv2.WINDOW_NORMAL)
        cv2.imshow('resultImage',img)
        cv2.resizeWindow('resultImage', 600,600)
        key = 1
        print('Press q to quit')
        while key !=ord('q'):       # Quit on pressing q
            key = cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    it = imageTransform()
    it.main(0,1)
    it.main(0,2)
    it.main(0,3)
    it.main(10,11)
    it.main(10,12)
    it.main(10,13)




def isinpoly():
    # ctr = sum(polypts)/polypts.shape[0] # centroid
    # polypts = np.vstack([polypts, polypts[0,:]])
    # for i in range(polypts.shape[0]-1):
    #         l = np.cross(polypts[i,:],polypts[i+1,:])    # line between two pts

#!/usr/bin/env python3
############################################################################
# Script for homework 3 ECE661
#   Has functions for:
# - finding homographies given sets of coordinates
# - finding homography using vanishing line
# - finding homography using 1-step method C∞

# Author: Mythra Balakuntala
############################################################################
import cv2, numpy as np, math, time, matplotlib
import time
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

class imageTr():
    def __init__(self):
        # Points
        # Ensure first point corresponds to origin in destination
        #   and points are in some cyclic order
        self.p = {1 : np.array([[1141,816,1],[1257,762,1],[1245,945,1],[1126,993,1]]),
                    2 : np.array([[236,57,1],[336,72,1],[335,277,1],[236,283,1]]),
                    11: np.array([[0,0,1],[60,0,1],[60,80,1],[0,80,1]]),
                    12: np.array([[0,0,1],[40,0,1],[40,80,1],[0,80,1]]),
                    13: np.array([[0,0,1],[28,0,1],[28,22,1],[0,22,1]]),
                    14: np.array([[0,0,1],[123,0,1],[123,63,1],[0,63,1]]),
                    3: np.array([[1246,966,1],[3302,454,1],[3124,2800,1],[368,2392,1]]),
                    4: np.array([[560,1832,1],[2132,728,1],[2962,1082,1],[1500,2626,1]]),
                  }
        # Create image names dict
        self.img_names = {1:'HW3Pics/1.jpg',
                     2:'HW3Pics/2.jpg',
                     11:'HW3Pics/11.jpg',
                     12:'HW3Pics/12.jpg',
                     3:'HW3Pics/3.jpg',
                     4:'HW3Pics/4.jpg',
                    13:'HW3Pics/13.jpg',
                    14:'HW3Pics/14.jpg',
                     }

    def cornerpts(self, img0_pth, H, p):
        # Computes the size of the projected image
        # Uses input image size and transform to return proj image size
        # Inputs: img0 - source image, H - homogrphy from source to dest
        # Outputs: w - width of dest image, h - height of dest image,
        #          H_new - new homography with added translation for (0,0)
        img0 = cv2.imread(img0_pth)
        src_h, src_w, = img0.shape[0:2] # source height and width
        p0 = np.array([src_w, src_h, 1] )
        # Find corresponding cornerpts on dest
        crnr_pts = np.zeros((4,3))
        src_pts = np.zeros((4,3))
        for i in range(4):
            src_pts[i,:] = np.multiply(np.array([i%2^int(i/2), int(i/2), 1]), p0)
            crnr_pts[i,:] = H@src_pts[i,:]
        # Normalize corner points max x3 = 1 for each point
        crnr_pts = np.divide(crnr_pts,crnr_pts[:,2][:,None])
        # Compute translation
        t = np.min(crnr_pts,0)
        T = np.hstack([np.eye(3,2), np.array([-t[0],-t[1],1])[:,None]])   # Translation homography
        # ^ modified so it can be added into H, i.e setting x3 = 0
        # so that H[2,:] - t (new translation) will have last coordinate as 1
        # compute size required for dest image
        w, h = (np.max(crnr_pts,0) - np.min(crnr_pts,0))[0:2]
        # Translate the corner Points
        crnr_pts = ((T@crnr_pts.T).astype(int)).T
        # Reform H adding the translation to H
        H_new = T@H
        return(int(h),int(w),crnr_pts,H_new,src_pts)

    def getProjImg(self,p0,p1,img0_pth,img1_pth, H):
        # Inputs: p0 - Source image reference points
        #         p1 - Destination image reference points
        #   img0_pth - Path to source image
        #   img1_pth - Path to destination image
        #          H - Homograpy from dest to src, calculated from p0 and p1 if not given
        # computes homography between the frames and maps img1 onto img2
        # Load the images

        img0 = cv2.imread(img0_pth)
        img1 = cv2.imread(img1_pth)
        # Limits of bounding box
        # Compute bounding box of distorted image (p1)
        clim = [min(p1[:,0]),max(p1[:,0])]
        rlim = [min(p1[:,1]),max(p1[:,1])]
        print("clim is ", clim)
        print("rlim is ", rlim)
        print("Projecting ..", end="", flush=True)
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
                try:
                    img1[j,i,:] = img0[projp[1],projp[0],:]
                except:
                    continue
            if i%100 == 0:
                print(".", end="", flush=True)
        print(" ")
        return img1


    def isinside(self,p, polypts):
        # Checks if given point is inside a polygon
        # polypts defines the vertices of the polygon in some cyclic order
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
        return(isin)


    def viewimg(self,img):
        # Function to view image
        cv2.namedWindow('resultImage',cv2.WINDOW_NORMAL)
        cv2.imshow('resultImage',img)
        cv2.resizeWindow('resultImage', 600,600)
        key = 1
        print('Press q to quit')
        while key !=ord('q'):       # Quit on pressing q
            key = cv2.waitKey(0)
        cv2.destroyAllWindows()

    def findHVline(self,p0):
        #-----------------------------------------------------------------------
        # Function to copmute vanising line given Points
        # Input : p0 - points defining two sets of parallel lines
        # The points are asssumed to be in order i.e. 1-2 and 3-4 form parallel lines
        #  2-3 and 4-1 form another set of parallel lines
        # Also the lines 1-2 and 2-3 are assumed to be perpendicur
        #   => Choose vertices of a rectangle in original image (undistorted)
        # Output: H - homography to remove projective
        #         l - array of lines (1-3 2-4 parallel, 1-2 3-4 perp )
        #------------------------------------------------------------------------
        # Find the 4 lines
        # append first point to end so we can cycle
        p0 = np.vstack([p0,p0[0,:]])
        l = np.zeros((4,3)) # array of lines
        for i in range(4):
            l[i,:] = np.cross(p0[i,:],p0[i+1,:])
        # Find points on vl
        vpt1 = np.cross(l[0,:],l[2,:])
        vpt2 = np.cross(l[1,:],l[3,:])
        vpt1= vpt1/vpt1[-1]
        vpt2= vpt2/vpt2[-1]
        # Find vanishing line
        vl = np.cross(vpt1,vpt2)
        vl = vl/vl[-1]
        # Homography to remove projective transform
        H = np.vstack([np.eye(2,3),vl])
        return(H,l)

    def findHaff(self, l):
        #-----------------------------------------------------------------------
        # Function to copmute affine transform given
        # Input : l - array of lines (1-2, 3-4  are pairwise perp )
        # Output : Homography to remove affine transform
        #------------------------------------------------------------------------
        # From cos(th) equations, we form equations for s in the form As = b
        A = np.zeros((2,2))
        A[0,:] = np.array([l[0,0]*l[1,0], l[0,0]*l[1,1] + l[0,1]*l[1,0]])
        A[1,:] = np.array([l[2,0]*l[3,0], l[2,0]*l[3,1] + l[2,1]*l[3,0]])
        b = np.array([-l[0,1]*l[1,1], -l[2,1]*l[3,1]])

        # Compute s vector
        s = np.linalg.pinv(A.T@A)@A.T@b
        smat = np.array([[s[0],s[1]],[s[1],1]]) # S matrix part of H
        # Eigen decomposition of smat
        w,v = np.linalg.eig(smat)
        # generate A matrix from S
        aMat = v@np.diag(np.sqrt(w))@v.T
        H = np.vstack([np.hstack([aMat,np.array([0,0])[:,None]]),np.array([0,0,1])])
        H = np.linalg.pinv(H)
        H = H/H[-1,-1]
        return(H)


    def perplines(self,pts):
        #-----------------------------------------------------------------------
        # Function to compute points for lines
        # Input : pts - vertices of reference rectangle
        # Output : perpendicular lines
        #------------------------------------------------------------------------
        l = np.ones((4,3))
        l[0,:] = np.cross(pts[0,:],pts[1,:])
        l[1,:] = np.cross(pts[1,:],pts[2,:])
        # Now for two other lines we construct a arificial rhombus
        #   and use its diagonals
        # First two pts are the same the bottom two points are computed from the length
        # dist between first two pts
        d = np.linalg.norm(pts[1,:] - pts[0,:])
        pt3 = pts[1,:] + d*(pts[2,:]-pts[1,:])/np.linalg.norm(pts[2,:]-pts[1,:])
        pt4 = pts[0,:] + d*(pts[3,:]-pts[0,:])/np.linalg.norm(pts[3,:]-pts[0,:])
        print(np.linalg.norm(pts[0,:] - pts[1,:]))
        print(np.linalg.norm(pts[0,:] - pt4))
        print(np.linalg.norm(pts[1,:] - pt3))
        print(np.linalg.norm(pt3 - pt4))
        # compute diagonal lines
        l[2,:] = np.cross(pts[0,:],pt3)
        l[3,:] = np.cross(pts[1,:],pt4)
        return(l)

    # The main function for problem 2
    def main(self, img_ind):
        # Main function computes given image and projected image
        #
        dimg_ind = img_ind + 10 # Destination image index
        # Compute homography to remove projective transform
        H, l = self.findHVline(self.p[img_ind])
        # Find boudary of dest_image
        h,w,crnr_pts, H_new,src_pts = self.cornerpts(self.img_names[img_ind], H, self.p[img_ind][0,:])
        # Create empty image
        dest_img = np.zeros((h,w))
        cv2.imwrite(self.img_names[dimg_ind],dest_img)
        # Get projected image
        dest_img = self.getProjImg(src_pts, crnr_pts, self.img_names[img_ind],self.img_names[dimg_ind],np.linalg.pinv(H_new))
        cv2.imwrite('p2_proj_modimg'+str(img_ind)+'.jpg',dest_img)

        # Get points for perpendicular lines
        ppts = (H@self.p[img_ind].T).T
        ppts = np.divide(ppts,ppts[:,2][:,None])
        l = self.perplines(ppts)
        # Find affine imageTransform
        Ha = self.findHaff(l)
        H_c = Ha@H # H combined from both projective and affine
        # Find boudary of dest_image
        h,w,crnr_pts, H_new,src_pts = self.cornerpts(self.img_names[img_ind], H_c, self.p[img_ind][0,:])
        # Create empty image
        dest_img = np.zeros((h,w))
        cv2.imwrite(self.img_names[dimg_ind],dest_img)
        # Get projected image
        dest_img = self.getProjImg(src_pts, crnr_pts, self.img_names[img_ind],self.img_names[dimg_ind],np.linalg.pinv(H_new))
        cv2.imwrite('p2_modimg'+str(img_ind)+'.jpg',dest_img)
        # Show image and destroy on 'q' pressed
        # uncomment to view
        #self.viewimg(dest_img)

if __name__ == "__main__":
    it = imageTr()
    it.main(1)  # For image 1.jpg
    it.main(2)  # For image 2.jpg
    it.main(3)  # For image 3.jpg
    it.main(4)  # For image 4.jpg

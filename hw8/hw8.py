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
from scipy.optimize import leastsq as lsq

# Callibration points class
class CameraCallib():
    def __init__(self, nimgs, dset = 1):
    #----------------------------------------------------------------------
    # Initialization
    # inputs: dset - dataset number
    # ---------------------------------------------------------------------
        self.img_path = 'Files/Dataset'+str(dset)+'/Pic_'   # Path to image dataset
        self.nimgs = nimgs
        self.nx = 8 # 2 * number of x squares
        self.ny = 10 # 2 * number of y squares
        self.kill = False
        self.dchk  = 1   # distance between points
        # [x,y] for point i is (i/10*self.dchk, i%10*self.dchk)
        self.crs = np.array([[ int(i/10)*self.dchk, int(i%10)*self.dchk ,1 ] for i in range(self.nx*self.ny)])

# Some basic functions -----------------------------------------------------
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

    # computes the least distance between p and pts
    def dist(self, pts, p, type=1):
        #type = 1 : 8 connected distance
        if type == 1:
            return(min(np.max(abs(pts - p),1)))
        #type = 2 : Euclidean distance
        elif type == 2:
            return(np.sqrt(min(np.sum((pts-p)**2,1))))

# function to compute the rotation matrix from rodigues
    def q_from_rod(self,r):
        th = np.linalg.norm(r)
        r = r/th
        k = np.cos(th)*np.eye(3) + (1 - np.cos(th))*r[:,None]@r[:,None].T \
        + np.sin(th)*np.array([[0,-r[2],r[1]],[r[2],0,-r[0]],[-r[1],r[0],0]])
        return(k)

# function to get rodrigues parameters from rotation matrix
    def q2rod(self, R):
        r = (((cv2.Rodrigues(R))[0]).T)[0]
        return(r)
#------------------------------------------------------------------------------
    # Get corner poitns for the callibration chekered board image.
    def get_corners(self, img_rgb, ind):
    #----------------------------------------------------------------------
    # Function to compute corner points in a callibration pattern
    #   Uses Canny and Hough to find corners, and LM to refine them.
    # inputs: img - image of callbiration pattern (Grayscale)
    # output: crnr_pts - 2 col Array containing corner points pixel positions
    #   1st col - x, 2nd col - y (Rows are arranged in order of corner numbers)
    # ---------------------------------------------------------------------
        img = cv2.cvtColor(img_rgb,cv2.COLOR_BGR2GRAY)  # GRAY
        img_hl = np.copy(img_rgb)  # Create copy of input image
        img_edg = cv2.Canny(img,100,500)    # Get edges from canny
        minLineLength = 10
        maxLineGap = 5
        # Compute hough lines
        lines = cv2.HoughLinesP(img_edg, 1, 1*np.pi/180, 15, minLineLength = 15, maxLineGap = 20)
        # print('Number of Hough lines = ',lines.shape[0])
        cv2.imwrite('rimgs/Pic_'+str(ind)+'_edg.jpg',img_edg)
        ci = 0
        crnr_pts = np.zeros((1,2))
        for x1,y1,x2,y2 in lines[:,0]:
            # img_hl = np.copy(img_rgb)
            cv2.line(img_hl,(x1,y1),(x2,y2),(255,0,0),1)
            # Avoid repetition, i.e. both horizontal and vertical lines
            #   give same corners so choose only corners from vertical ..
            if abs(abs(np.arctan2(y2-y1,x2-x1)) - np.pi/2) < np.pi/4 :
                # Check if points is already in corners
                if self.dist(crnr_pts, [x1,y1], 1) > 15 and self.dist(crnr_pts, [x2,y2], 1) > 15:
                    # ^ Check if any point is closer than 10 pixels, if so its the same
                    crnr_pts = np.vstack((crnr_pts,np.array([x1,y1])))  # Store corners
                    crnr_pts = np.vstack((crnr_pts,np.array([x2,y2])))
                    # cv2.circle(img_hl,(x1,y1),1,(0,255,0),1) # Draw corners
                    # cv2.circle(img_hl,(x2,y2),1,(0,255,0),1)
                    # self.viewimg(img_hl,1000)
        crnr_pts = crnr_pts[1:,:].astype(np.intp)   # Eliminate first row (init row)
        spts = crnr_pts.astype(np.float32).reshape(-1,1,2)
        # Refine corners based on subpixel estimate
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.5)
        spts = cv2.cornerSubPix(img, spts, (15, 15), (-1, -1), term)
        spts = spts[:,0].astype(np.int)
        # print('Number of corner points found = ', crnr_pts.shape[0])
        # get sorted points
        spts, xint = self.order_pts(spts)
        spts = spts.astype(int)
        j = 0
        for x1,y1 in spts:
        # cv2.drawContours(img_hl, [np.int0(bpts)],0 , (0,0,255),2)
            cv2.putText(img_hl, str(j), (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255))
            cv2.circle(img_hl,(x1,y1),1,(0,255,0),1)
            j +=1
            # if j%1 ==0:
            #     self.viewimg(img_hl,1000)
            #     print(x1,y1)
            #     print(xint[int(j/10 - 1)*5:int(j/10)*5])
        # Eliminate duplicates ... we get two points for each.
        cv2.imwrite('rimgs/Pic_'+str(ind)+'_hl.jpg',img_hl)
        return(spts)

# functino to order points from top left to bottom right
    def order_pts(self, pts):
    #----------------------------------------------------------------------
    # Function to order of points and the boundary points
    # Input: pts - points to be sorted.
    # pts should be of form rows pts_i and pts_i+1 form vertical lines i=2j, j=0,1,..
    # Output: opts - sorted points rowwise and column wise in each row
    #       bpts - 4 corner points
    # ---------------------------------------------------------------------
        ipts = np.copy(pts.reshape(-1,4))   # input pts
        bpts = np.zeros((4,2),dtype=np.intp)
        opts = np.copy(pts)
        # Find leftmost and rightmost point
        bpts[(0,2),:] = pts[np.argsort(pts[:,0]),:][(0,-1),:]
        # Find topmost and bottom most point
        bpts[(1,3),:] = pts[np.argsort(pts[:,1]),:][(0,-1),:]
        # return(bpts)
        # compute x intercepts
        xint = np.zeros(int(pts.shape[0]/2))
        for i in np.arange(0,pts.shape[0],2):
            xint[int(i/2)] = np.linalg.det(pts[i:i+2,:])/(pts[i,1]-pts[i+1,1])

        # Sort accoding to x intercepts
        ipts = ipts[np.argsort(-xint),:].reshape(-1,2)
        xint = xint[np.argsort(xint)]
        for j in range(self.nx):
            tmp = ipts[j*self.ny:(j+1)*self.ny,:]
            ipts[j*self.ny:(j+1)*self.ny,:] = tmp[np.argsort(tmp[:,1]),:]
        return(ipts, xint)

# function to compute K from W
    def kfromw(self,W):
    #----------------------------------------------------------------------
    # Function to compute K from W = K^-T * K^-1
    # Input: matrix W
    # Output: camera intrinsic parameter matrix
    # ---------------------------------------------------------------------
        x0 = (W[0,1]*W[0,2] - W[0,0]*W[1,2])/(W[0,0]*W[1,1]-W[0,1]*W[0,1])
        lam = W[2,2] + (W[0,2]*W[0,2] + x0*(W[0,1]*W[0,2] - W[0,0]*W[1,2]))/W[0,0]
        ax = np.sqrt(lam/W[0,0])
        ay = np.sqrt((lam*W[0,0])/(W[0,0]*W[1,1]-W[0,1]*W[0,1]))
        s = -(W[0,1]*ax*ax*ay)/(lam)
        y0 = s*x0/ay - W[0,2]*ax*ax/lam
        return(np.array([[ax,s,x0],[0,ay,y0],[0,0,1]]))

    # function to compute intrinsic parameter matrix K
    def findk(self):
        icrs = np.ones((self.nx*self.ny,3,self.nimgs))# image corners
        h = np.zeros((3,3,self.nimgs))# image corners
        vmat = np.zeros((2*self.nimgs, 6))
        # Define vij
        vf = lambda i,j,h: np.array([h[0,i]*h[0,j], h[0,i]*h[1,j] + h[1,i]*h[0,j],\
          h[1,i]*h[1,j], h[2,i]*h[0,j] + h[0,i]*h[2,j], h[2,i]*h[1,j]\
          + h[1,i]*h[2,j], h[2,i]*h[2,j]])
        # print(self.crs)
        # Compute Vmat using all images
        for i in range(self.nimgs):
            # print(self.img_path+str(i+1)+'.jpg')
            img_rgb = cv2.imread(self.img_path+str(i+1)+'.jpg')  # RGB
            # Get corners using Canny and Hough
            icrs[:,0:2,i] = self.get_corners(img_rgb,i+1)
            # Find homography
            h[:,:,i], mask = cv2.findHomography(self.crs, icrs[:,:,i], cv2.RANSAC,3.0)
            # Fill V matrix...
            vmat[2*i:2*(i+1),:] = np.array([vf(0,1,h[:,:,i]),vf(0,0,h[:,:,i])-vf(1,1,h[:,:,i])])
        u,s,vh = np.linalg.svd(-vmat)    # Solve Vw = 0 using SVD
        wsol = (vh.T)[:,-1]             # Null space solution
        # compute K from wsol = K^-T * K^-1
        w = np.array([[wsol[0],wsol[1],wsol[3]],[wsol[1],wsol[2],wsol[4]]\
        ,[wsol[3],wsol[4],wsol[5]]])
        e = np.linalg.eig(w)
        l = np.linalg.cholesky(w)
        kc = np.linalg.pinv(l.T)
        kc = kc/kc[-1,-1]
        k = self.kfromw(w)
        return(h,icrs,k,kc)

# function to compute the extrinsic parameters
    def find_ext(self, k, h):
    #-
    #-
        r = np.zeros((3,3))
        ki = np.linalg.pinv(k)
        lam = 1/np.linalg.norm(ki@h[:,0])
        r[:,0] = lam * ki@h[:,0]
        r[:,1] = lam * ki@h[:,1]
        r[:,2] = np.cross(r[:,0],r[:,1])
        t = lam*ki@h[:,2]
        return(r,t)

# The cost function for LM refining of Camera projection matrix P (K,r,t)
    def cp_lm(self, k):
        # find rotation matrix
        kmat = np.array([[k[0],k[1],k[2]],[0,k[3],k[4]],[0,0,1]])
        npts = self.nx*self.ny
        res = np.zeros(2*self.nimgs*npts)
        m = 5+3*self.nimgs
        for j in range(self.nimgs):
            # Back project points
            rmat = self.q_from_rod(k[5+j*3:5+(j+1)*3])
            t = k[m+j*3:m+(j+1)*3]
            ppts = (kmat@(np.append(rmat[:,0:2],t[:,None],1))@(self.crs.T))  # projected points
            ppts = (ppts/ppts[-1,:]).T
            dx = (self.icrs[:,:,j] - ppts)[:,0:2]
                # print(dx)
            res[2*j*npts:2*(j+1)*npts] = dx.reshape(-1)
        return(res)

# The main function
    def main(self):
        # Get inital estimates for K
        h, self.icrs, k, kc = self.findk()
        # Back project for few images
        for i in range(4):
            img_rgb = cv2.imread(self.img_path+str(i+1)+'.jpg')  # RGB
            # Find back projected points
            r,t = self.find_ext(k,h[:,:,i])
            ppts = (k@np.append(r[:,0:2],t[:,None],1)@(self.crs.T))  # projected points
            ppts = ((ppts/ppts[-1,:]).T).astype(np.int)
            j = 0
            for x1,y1 in ppts[:,0:2]:
            # cv2.drawContours(img_hl, [np.int0(bpts)],0 , (0,0,255),2)
                cv2.putText(img_rgb, str(j), (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255))
                cv2.circle(img_rgb,(x1,y1),1,(0,255,0),1)
                j+=1
            cv2.imwrite('rimgs/Pic_'+str(i+1)+'_bp.jpg',img_rgb)
        # ---------------------------------------------------------------------
        # LM to refine parameters
        self.h = h
        r0 = np.zeros((3,self.nimgs))
        t0 = np.zeros((3,self.nimgs))
        for i in range(self.nimgs):
            # Find r and t
            rmat, t0[:,i] = self.find_ext(k,h[:,:,i])
            r0[:,i] = self.q2rod(rmat)
            if i < 4:
                print( r0[:,i], t0[:,i])
        # Create init vector for LM
        v0 = np.zeros(5+6*self.nimgs)
        # The k part - 5 params
        v0[0:5] = np.array([k[0,0],k[0,1],k[0,2],k[1,1],k[1,2]])
        v0[5:5+3*self.nimgs] = r0.T.reshape(-1)
        v0[5+3*self.nimgs:5+6*self.nimgs] = t0.T.reshape(-1)
        s = ls(self.cp_lm, v0, method='lm',max_nfev = 2)
        kmat = np.array([[s.x[0],s.x[1],s.x[2]],[0,s.x[3],s.x[4]],[0,0,1]])
        print(kmat)
        print(k)
        m = 5+3*self.nimgs
        print(s.x[5:8], s.x[m:m+3])
    #-------------------------------------------------------------------------
        # Back project for few images after LM
        for i in range(4):
            img_rgb = cv2.imread(self.img_path+str(i+1)+'.jpg')  # RGB
            # Find back projected points
            rmat = self.q_from_rod(s.x[5+i*3:5+(i+1)*3])
            t = s.x[m+i*3:m+(i+1)*3]
            ppts = (kmat@np.append(r[:,0:2],t[:,None],1)@(self.crs.T))  # projected points
            ppts = ((ppts/ppts[-1,:]).T).astype(np.int)
            j = 0
            for x1,y1 in ppts[:,0:2]:
            # cv2.drawContours(img_hl, [np.int0(bpts)],0 , (0,0,255),2)
                cv2.putText(img_rgb, str(j), (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255))
                cv2.circle(img_rgb,(x1,y1),1,(0,255,0),1)
                j+=1
            cv2.imwrite('rimgs/Pic_'+str(i+1)+'_lm.jpg',img_rgb)

if __name__ == '__main__':
    cc = CameraCallib(40,1)  # number of image to use and dataset no. (1)
    cc.main()

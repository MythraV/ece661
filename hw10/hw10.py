#!/usr/bin/env python3
############################################################################
# Script for homework 10 ECE661, part 1
# Contains pattern classification class with the following functions
#  - PCA: Principal component analysis
#  - LDA: Linear Discriminant analysis
#  - kNN (k-Nearest Neighbours) classifier
# The class is written to be independent of any homework specific values or
#    variables, so as to be useful in any other codes. 
# Author: Mythra Balakuntala
############################################################################
import numpy as np, cv2
from scipy.optimize import least_squares as ls
from matplotlib import pyplot as plt
class PatternClassify():
    def pca(self, x, p):
    #-----------------------------------------------------------------------
    # Function to do pca and find eigen space for input vectors in x
    # Inputs: x - matix with each column being a normalized vectorized data
    #   p - number of eigenvectors to consider
    # Outputs: y - input feature matrix, w - eigen vectors for the space,
    #    m - mean of input space
    #-----------------------------------------------------------------------
        # n = x.shape[1]  # number of images
        m = (x.T).mean(0)
        x = (x.T - m).T   # Subtract mean
        xtx = x.T@x # Compute eigvecs u for x^T x aand eigvecs of C will be X*u
        e, u = np.linalg.eig(xtx)
        # Sort vectors according to descending order of e
        #   e(all positive as xtx is pos semi defininte)
        u = u[:,e.argsort()[::-1]]
        w = (x@u)[:,0:p]    # Extract p largest eigen vectors
        w = w/np.linalg.norm(w,axis=0)
        y = w.T@x   # Get feature vectors
        return(y,w,m)  # Normalized eigen vecs for C

    def lda(self, x, p, nimgs, dthres = 1e-5):
    #-----------------------------------------------------------------------
    # Function to do lda and find eigen space for input vectors in x
    # Inputs: x - matix with each column being a normalized vectorized data
    #   p - number of eigen vectors needed, nimgs - number of images per class
    #   dthres - theshold to discard eigen values
    # Outputs: w - eigen vectors for the space, m - mean of input space
    #-----------------------------------------------------------------------
        n = x.shape[1]  # number of images
        m = (x.T).mean(0)   # global mean
        mi = [(x[:,i:i+nimgs].T).mean(0) for i in np.arange(0,n,nimgs)] #class means
        mc = np.array(mi)
        # Solve for eigen vecs of sb = sum_i(mi-m)(mi-m)^T, M = (m1-m|m2_m|m3_m..)
        mmat = (mc - m).T
        e_t, ut = np.linalg.eig((mmat.T)@mmat)
        u = ut[:,np.argsort(e_t)[::-1]]
        e_sb = e_t[np.argsort(e_t)[::-1]]
        # Discard eigen values close to 0
        idx = np.where(e_sb > dthres)[0]
        e_sb = e_sb[idx]
        u = u[:,idx]
        v =  mmat@u # Compute Eigen vecs of SB
        v = v/np.linalg.norm(v,axis=0)  #normalize
        # Comput Eigvecs for Sw (with class scatter)
        db2 = np.linalg.pinv(np.diag(np.sqrt(e_sb)))
        z = v@db2
        xw = np.copy(x)  # Matrix xw = [x1-m1|x2-m1|...|xm-m1|xm+1-m2...] where m is
        # number of images per class
        for i in range(n):
            xw[:,i] = x[:,i]-mc[int(i/nimgs)]
        zmat = z.T@xw
        e_sw, u_sw = np.linalg.eig(zmat@zmat.T)
        u_sw = u_sw[:,np.argsort(e_sw)]    # sort in ascending order
        u_sw = u_sw[:,0:p]  # Extract p smallest eigen vectors
        w = z@u_sw # Egenvectors for lda
        w = w/np.linalg.norm(w,axis=0)  # normalize
        x = (x.T - m).T   # Subtract mean
        y = w.T@x   # Get feature vectors
        return(y,w,m)  # Normalized eigen vecs for C

    def knn(self, y, w, m, x_t, nimgs, k=1):
    #-----------------------------------------------------------------------
    # Function to classify based on k nearest neighnours
    # Inputs: y - trained feature values, w - basis/eigen vectors
    #   m - mean to subtract, k - number of nearest neighbours to vote
    #   x_t - matrix of normalized test vectors,
    #   nimgs - number of images per class(train set)
    # Outputs: tst_lbl - classified labels
    #-----------------------------------------------------------------------
        x_t = (x_t.T - m).T # subtract means
        y_t = w.T@x_t   # Compute feature values for test dataset
        # print(y_t[:,0])
        tst_lbl = np.zeros(y_t.shape[1]).astype(int)
        ncls = int(y.shape[1]/nimgs)
        tr_lbl = np.repeat(range(ncls),nimgs) # Training class labels
        # Compute nearest neighbours
        for i in range(y_t.shape[1]):
            d = np.linalg.norm((y.T-y_t[:,i]).T,axis=0)
            knn = tr_lbl[d.argsort()][0:k]  # Get k nearest neighbours
            tst_lbl[i] = np.argmax(np.bincount(knn))  # Most frequent label is detected label
        return(tst_lbl)

    def get_acc(self, tru_lbl, tst_lbl):
        #-----------------------------------------------------------------------
        # Function to compute accuracies
        # Inputs: tru_lbl - true labels of the images
        #   tst_lbl - detected labels for the images
        # Outputs: acc - accuracy
        #-----------------------------------------------------------------------
        nTrue = np.sum(tst_lbl == tru_lbl)
        return(nTrue/tru_lbl.shape[0])

# function to vectorize nd data
    def vectorize(self, fpath, nimgs, ncls, ftype='png'):
    #-----------------------------------------------------------------------
    # Function to vectorize images in a folder (RGB images are converted to gray)
    # Inputs: fpath - path to folder, nimgs - number of images per class,
    #   ncls - number of classes
    # Outputs: x - Matrix of vectorized images, each column is vector
    #   corresponding to one vectorized nomalized image
    # Format for images -- fpath/clsid_imgid.ftype ex: ~/Data/02_13.png
    #-----------------------------------------------------------------------
        imat = []
        n = nimgs*ncls # Total number of images
        for i in range(1,ncls+1):
            for j in range(1,nimgs+1):
                # Read as gray scale image
                img = cv2.imread(fpath+str(i).zfill(2)+'_'+str(j).zfill(2)\
                +'.'+ftype, cv2.IMREAD_GRAYSCALE)
                ivec = np.copy(img.reshape(-1))
                imat.append(ivec)
        x = np.array(imat,dtype=np.float).T
        x = x/np.linalg.norm(x,axis=0)
        return(x)

def main():
    #--------------------------------------------------------------------------
    # Main function for hw10 part 1
    # Computes and evaluates PCA and LDA classifiers for given face dataset
    #--------------------------------------------------------------------------
    # User Vars
    # file paths (Should contain 2 folders train and test with images)
    #   Format for images -- fpath/clsid_imgid.ftype ex: ~/Data/02_13.png
    fpath = '/home/crl/Data/ECE661_2018_hw10_DB1/'
    ftype = 'png'
    ncls = 30   # Number of classes for training data
    nimgs = 21  # number of images per class in training data
    p = 13  # number of pricipal eigen vectos to use
    ncls_tst = 30   # Number of classes for test data
    nimgs_tst = 21  # number of images per class in test data
    #--------------------------------------------------------------------------
    #----- Training ---
    pc = PatternClassify()
    # Train images
    x = pc.vectorize(fpath+'train/',nimgs, ncls, ftype)
    acc_pca = np.zeros(20)
    acc_lda = np.zeros(20)
    print('Computing.',end='',flush=True)
    for p in range(20):
        y_pca, w_pca, m_pca = pc.pca(np.copy(x),p)   # normalized eigen vectors from PCA
        y_lda, w_lda, m_lda = pc.lda(np.copy(x),p,nimgs)
        #--------------------------------------------------------------------------
        # --- Testing ---
        # Load test images and normalize
        x_t = pc.vectorize(fpath+'test/',nimgs_tst, ncls_tst, ftype)
        tru_lbl = np.repeat(range(ncls_tst),nimgs_tst) # Actual labels for test images
        # Get accuracies
        cl_pca = pc.knn(y_pca, w_pca, m_pca, x_t, nimgs)
        acc_pca[p] = pc.get_acc(tru_lbl, cl_pca)
        cl_lda = pc.knn(y_lda, w_lda, m_lda, x_t, nimgs)
        acc_lda[p] = pc.get_acc(tru_lbl, cl_lda)
        print(".",end='',flush=True)
    p1 = plt.plot(acc_pca,'bo',acc_pca,'b')
    p2 = plt.plot(acc_lda,'ro',acc_lda,'-r')
    plt.plot(np.ones(20),'k--')
    plt.ylabel('Accuracy')
    plt.xlabel('p value')
    plt.legend((p1[0],p2[0]),('PCA', 'LDA'))
    plt.show()

if __name__ == '__main__':
    main()

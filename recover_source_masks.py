import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

import os 

import functools 
#from skimage.measure import block_reduce
import cv2 as cv2 
import scipy as sp
from scipy.signal import welch
import time 
#import torch
from concurrent import futures
#import itertools
from itertools import repeat 


import process_TM as TM
#import explore_data as explore 
#import check_images as check

def getA():

    direc = "/local-storage/projects/foundation/waveguide/"
    direc_img = direc + "july_2025/"
    
    file_list = [f for f in os.listdir(direc_img) if os.path.isfile(os.path.join(direc_img,f))]
    print("num files ",len(file_list))
   # print(file_list[:20])
    file_list = sorted(file_list,key=functools.cmp_to_key(TM.greater_filev2))  
    print(file_list[0])            
    #A = TM.getA_from_files(direc,file_list,save_file=True,out_path = direc + "Anewest.npy")
    A = []
    return A

def recover_mask_with_pinv(A,Apinv,file_path="",H=151,W=153):

    Y = np.loadtxt(file_path)
    print(" Y ",Y.shape, Y.dtype)
    ry,cy = Y.shape
    Y = Y.astype(np.float32)
    plt.imshow(Y)
    
    rA,cA = A.shape
    print(" A ", rA, cA)
    y = np.reshape(Y,(rA,1)) 
    x = np.dot(Apinv,y)

    H = 151
    W = 153
    X = np.reshape(x,(H,W)) # ,order='F')
    plt.figure()
    plt.imshow(X)
    
    
    plt.show()
    

def main():
    #direc = "/home/zunigac/Projects/MLWaveguide/"
    direc = "/local-storage/projects/foundation/waveguide/"
    A = np.load(direc + "A288x384_july2025.npy")
    A = A.astype(np.float32)
    Apinv = np.load(direc + "A288x384pinv_july2025.npy")
    
    file_name =  "slit_centered.txt"
    file_path = "july_2025/" + file_name
    recover_mask_with_pinv(A,Apinv,file_path)
    Ap = getA()
    
    
    
    
    

if __name__ == "__main__":
    main()

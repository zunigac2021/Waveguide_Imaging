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
import simulate_masks as masks

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
    
def getA_pinv(A,file_name=""):
    """
    Compute the Moore-Penrose pseudoinverse of a matrix A and optionally save it to a file.

    Parameters:
    A (numpy array): The input matrix.
    file_name (str, optional): The file name to save the pseudoinverse to. If empty, the pseudoinverse is not saved. Defaults to "".

    Returns:
    Apinv (numpy array): The Moore-Penrose pseudoinverse of A.
    """
    Apinv = np.linalg.pinv(A)
    if len(file_name) >0 :
        np.save(file_name,Apinv)
    return Apinv
    

def recover_mask_with_pinv(A,Apinv,Y=None,file_path="",H=151,W=153):
    """
    Recover a mask from a given matrix A, its pseudoinverse Apinv, and a file path.

    Parameters:
    A (numpy array): The original matrix.
    Apinv (numpy array): The pseudoinverse of matrix A.
    file_path (str, optional): The path to the file containing the data. Defaults to "".
    H (int, optional): The height of the output image. Defaults to 151.
    W (int, optional): The width of the output image. Defaults to 153.

    Returns:
    None
    """
    if Y is None and len(file_path)>0:
        print("loading file")
        Y = np.loadtxt(file_path)
        print("Y min max ",np.min(Y), np.max(Y))
    print(" Y ",Y.shape, Y.dtype)
    ry,cy = Y.shape
    Y = Y.astype(np.float32)
    #Y2 = Y*Y
    plt.imshow(Y)
    plt.title("Image")
    
    rA,cA = A.shape
    print(" A ", rA, cA)
    y = np.reshape(Y,(rA,1)) 
    x = np.dot(Apinv,y)

    
    X = np.reshape(x,(H,W)) # ,order='F')
    plt.figure()
    plt.imshow(X)
    plt.title("Recovered Mask")
    
    plt.show()
    

def main():
    #direc = "/home/zunigac/Projects/MLWaveguide/"
    direc = "/local-storage/projects/foundation/waveguide/"
    A = np.load(direc + "A288x384hp_july2025.npy")
    A = A.astype(np.float32)
    
    print("A min max ", np.min(A),np.max(A))
    Apinv = np.load(direc + "A288x384hppinv_july2025.npy")
    
    file_name =  "slit_centered.txt" #"no_screen.txt" # "background.txt" # 
    file_path = "july_2025/" + file_name
    #Ap = getA()
    H,W = 151,153
    Hi,Wi = 288,384
    num_source_pixels = H*W
    mask = masks.get_mask(H,W,"3-slit") # "circular",{"R":H//2})
    mask_v = np.reshape(mask,(num_source_pixels,1))
    
    #plt.imshow(mask)
    #plt.title("Original Mask")
    ysim = np.dot(A,mask_v)
    print("ysim min max ",np.min(ysim),np.max(ysim))
    Ysim = np.reshape(ysim,(Hi,Wi))
    plt.figure()
    #plt.imshow(Ysim)
    #plt.figure()
    
    recover_mask_with_pinv(A,Apinv,file_path=file_path) 
    plt.show()
    
    
    
    
    
    

if __name__ == "__main__":
    main()

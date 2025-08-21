import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

import os 
import re

import functools 
#from skimage.measure import block_reduce

import time 
from concurrent import futures
from itertools import repeat 
import scipy as sp
import cv2 as cv2
from scipy.ndimage import convolve1d
from scipy.signal import firwin, welch





def filter_image_hplp(img,distortion_freq,eps=0.03,num_taps=65):

    hpf = firwin(num_taps, distortion_freq - eps,
               pass_zero='highpass', fs=1)
    lpf = firwin(num_taps, eps, pass_zero='lowpass', fs=1)
    img_filtered =  convolve1d(convolve1d(img, hpf, axis=0), lpf, axis=1)
    #img_filtered =  convolve1d(img, hpf, axis=1)
    
    return img_filtered
    
def filter_image_ps(img,file_path=""):
    # following https://stackoverflow.com/questions/65480162/how-to-remove-repititve-pattern-from-an-image-using-fft

   
    if len(file_path)>0:
        img = np.loadtxt(file_path)
    Wy,Hy = img.shape # 384,288
    
    num_pixels = Wy*Hy
    Wyh, Hyh = Wy//2, Hy//2
    #plt.imshow(img) # ,cmap='gray')
    #plt.title("Original image 3-slit centered")
    
    img0 = np.mean(img,axis=0)       
    fx, Sx = welch(img0,nperseg=64)   
    plt.plot(fx,Sx)
     
    Sx[fx<1/25] = 0
    distortion_freq = fx[Sx.argmax()]
    print("dist freq ",distortion_freq)
    img_filt = filter_image_hplp(img,distortion_freq)
    return img_filt
    
    
def main():
    print("main")
    
    
if __name__ == "__main__":
    main()

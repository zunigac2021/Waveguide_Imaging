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

pi = np.pi

def merge_pixels(img,w=2,h=2):

    R,C = img.shape
    R2 = R//h
    C2 = C//w
    img_merged= np.zeros((R2,C2))
    for r in range(R2):
        sr = 2*r
        er = 2*(r+1)
        for c in range(C2):
            sc = 2*c
            ec = 2*(c+1)
            img_merged[r,c] = np.sum(img[sr:er,sc:ec])
    return img_merged

def filter_image_hplp(img,distortion_freq,eps=0.03,num_taps=65):

    hpf = firwin(num_taps, distortion_freq - eps,
               pass_zero='highpass', fs=1)
    lpf = firwin(num_taps, eps, pass_zero='lowpass', fs=1)
    #img_filtered =  convolve1d(convolve1d(img, hpf, axis=0), lpf, axis=1)       # seems to work best but not clear why
    #img_filtered =  convolve1d(img, lpf, axis=0)
    #img_filtered = convolve1d(img_filtered,hpf,axis=1)
    
    return img_filtered
    

def filter_image_hplpv2(img,distortion_freq,eps=0.03,num_taps=65):

    hpf = firwin(num_taps, eps,
               pass_zero='highpass', fs=1)
    lpf = firwin(num_taps, distortion_freq-eps, pass_zero='lowpass', fs=1)
    bpf = firwin(num_taps, [distortion_freq-eps,distortion_freq+eps], pass_zero='bandstop', fs=1)
    
    #img_filtered =  convolve1d(convolve1d(img, hpf, axis=0), lpf, axis=1)       # seems to work best but not clear why
    img_filtered =  convolve1d(img, bpf, axis=1)       # seems to work best but not clear why
    
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
    #plt.figure()
    #plt.plot(img0)
    #plt.title("Image Average along Rows")  
    fx, Sx = welch(img0,nperseg=64)
    plt.figure()   
    plt.plot(fx,Sx)
    plt.title("Power Spectrum of Image Averaged along Rows")  
    Sx[fx<0.01] = 0      # don't consider DC
    distortion_freq = fx[Sx.argmax()]
    print("dist freq ",distortion_freq)
    print("dist freq index ", 2*Hy*distortion_freq )
    img_filt = filter_image_hplpv2(img,distortion_freq,eps=0.01)
    return img_filt
    
def filter_freq_mask(img):
    H,W = img.shape
    plt.figure()
    img_fm0 = np.mean(img_fm,axis=0)
    plt.plot(img_fm0)
    plt.title("Mean 2D FFT along columns (axis=0)")
    
    ifx = np.argmax(img_fm0)
    print("ifx ",ifx, " max ",img_fm0[ifx],img_fm0[W-ifx])
    Wf = np.ones((H,W))
    Wf[:1,ifx] = 0
    Wf[:1,W-ifx] = 0
    img_fw = np.multiply(img_f,Wf)
    img_filt = np.abs(np.fft.ifft2(img_fw))
    return img_filt

def filter_line_simple(img):
 
    H,W = img.shape
    img0 = np.mean(img,axis=0)
    a = img0[0]
    #imgm = np.min(img,axis=0)
    img_filt = img - (img0-a)
    
    return img_filt
    
    

def check_filter():

    W, H = 384, 288
    Wh, Hh = W//2, H//2
    x = np.arange(W)
    y = np.arange(H)
    Px = 20
    Py = 30
    px = np.sin(2*pi*x/Px)
    py = np.cos(2*pi*y/Py)
    py = np.reshape(py,(H,1))
    img = np.tile(px,(H,1))
    #img = np.tile(py,(1,W))
       
    img[(Hh-50):(Hh+50),(Wh-50):(Wh+50)] =0.5
    img_f = np.fft.fft2(img)
    #print("fft2 shape ",img_f.shape)
    img_fm = np.abs(img_f)
    #plt.figure()
    #plt.imshow(img_fm[:20,:Wh])
    #plt.title("2D FFT of square with vertical lines (Zoomed to near 0 frequency)")
    
    
    #plt.imshow(img)                 
    
    img_filt = filter_image_ps(img)
    plt.figure()
    plt.imshow(img_filt)
    plt.show()

def test_conv1d():
    x = np.arange(20)
    X = np.reshape(x,(5,4))
    print(X)
    w = np.ones(2)
    Xc = convolve1d(X,w,axis=0)          # filter applied independently along each column
    print(Xc.shape)
    print(Xc)
        

def rof_denoise(im,U_init,tolerance=0.1,tau=0.125,tv_weight=100):
    """ From: https://github.com/trenton3983/Programming_Computer_Vision_with_Python/blob/master/PCV/tools/rof.py
        An implementation of the Rudin-Osher-Fatemi (ROF) denoising model
        using the numerical procedure presented in Eq. (11) of A. Chambolle
        (2005). Implemented using periodic boundary conditions.
        
        Input: noisy input image (grayscale), initial guess for U, weight of 
        the TV-regularizing term, steplength, tolerance for the stop criterion
        
        Output: denoised and detextured image, texture residual. """
        
    m,n = im.shape #size of noisy image

    # initialize
    U = U_init
    Px = np.zeros((m, n)) #x-component to the dual field
    Py = np.zeros((m, n)) #y-component of the dual field
    error = 1 
    
    while (error > tolerance):
        Uold = U
        
        # gradient of primal variable
        GradUx = np.roll(U,-1,axis=1)-U # x-component of U's gradient
        GradUy = np.roll(U,-1,axis=0)-U # y-component of U's gradient
        
        # update the dual varible
        PxNew = Px + (tau/tv_weight)*GradUx # non-normalized update of x-component (dual)
        PyNew = Py + (tau/tv_weight)*GradUy # non-normalized update of y-component (dual)
        NormNew = np.maximum(1,np.sqrt(PxNew**2+PyNew**2))
        
        Px = PxNew/NormNew # update of x-component (dual)
        Py = PyNew/NormNew # update of y-component (dual)
        
        # update the primal variable
        RxPx = np.roll(Px,1,axis=1) # right x-translation of x-component
        RyPy = np.roll(Py,1,axis=0) # right y-translation of y-component
        
        DivP = (Px-RxPx)+(Py-RyPy) # divergence of the dual field.
        U = im + tv_weight*DivP # update of the primal variable
        
        # update of error
        error = np.linalg.norm(U-Uold)/np.sqrt(n*m);
        
    return U,im-U # denoised image and texture residual

def filterA(A,H,W):

    rA,cA = A.shape
    Afilt = np.zeros((rA,cA))
    for c in range(cA):
        img = A[:,c]
        img = np.reshape(img,(H,W))
        img_filt = filter_line_simple(img)
        Afilt[:,c] = np.reshape(img_filt,(rA))
    return Afilt
    
    
    
def main():
    print("main")
   
    check_filter()
    #test_conv1d()
    
if __name__ == "__main__":
    main()

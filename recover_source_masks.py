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


import proces_TM as TM
import explore_data as explore 
import check_images as check

def getA():

    direc = "/local-storage/projects/foundation/waveguide/"
    direc_img = direc + "july_2025/"
    
    file_list = [f for f in os.listdir(direc_img) if os.path.isfile(os.path.join(direc_img,f))]
    print("num files ",len(file_list))
   # print(file_list[:20])
    file_list = sorted(file_list,key=functools.cmp_to_key(explore.greater_filev2))              # TODO move greater_filev2() to TM
    A = TM.getA_from_files(direc,file_list,save_file=True,out_path = direc + "Anewest.npy")
    return A




def main():
    A = np.load(direc + "A288x384hp_july2025.npy")
    A = A.astype(np.float32)
    
    

if __name__ == "__main__":
    main()

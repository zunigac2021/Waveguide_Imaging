import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

import os 
import re
#import check_images as check
import functools 
#from skimage.measure import block_reduce
import cv2 as cv2 
import time 
from concurrent import futures
from itertools import repeat 



def greater_filev2(file1,file2):
    """
    Compare two files based on the x and y coordinates extracted from their contents.

    The function uses regular expressions to extract the x and y coordinates from the file contents.
    It then compares the x coordinates, and if they are equal, compares the y coordinates.
    The function returns 1 if file1 has a greater coordinate, -1 if file2 has a greater coordinate, and 0 if the coordinates are equal.

    Args:
        file1 (str): The content of the first file.
        file2 (str): The content of the second file.

    Returns:
        int: 1 if file1 has a greater coordinate, -1 if file2 has a greater coordinate, and 0 if the coordinates are equal.
    """
    

    match1 = re.search(r"x=([-+]?[0-9]+(\.[0-9]+)?)(?:\s*)mm,\s*y=([-+]?[0-9]+(\.[0-9]+)?)(?:\s*)mm", file1)
    if match1:
        x1 = float(match1.group(1))
        y1 = float(match1.group(3))
        #print(x1,y1)
    match2 = re.search(r"x=([-+]?[0-9]+(\.[0-9]+)?)(?:\s*)mm,\s*y=([-+]?[0-9]+(\.[0-9]+)?)(?:\s*)mm", file2)
    if match2:
        x2 = float(match2.group(1))
        y2 = float(match2.group(3))
        #print(x2,y2)
    if x1 > x2:
        return 1
    elif x1 < x2:
        return -1
    else:
        if y1 > y2 :
            return 1
        elif y1 < y2:
            return -1
        else:
            return 0

def getA_from_files(direc,file_list,save_file=False,out_path = ""):
    """
    Loads data from a list of files in a specified directory and constructs a matrix A. 

    Parameters:
    direc (str): The directory path where the files are located.
    file_list (list): A list of file names to be loaded. Expected to be sorted in appropriate manner (incr x, incr y)
    save_file (bool, optional): If True, saves the constructed matrix A to a file. Defaults to False.
    out_path (str, optional): The path where the matrix A will be saved if save_file is True. Defaults to "".

    Returns:
    A (numpy array): A matrix where each column corresponds to a file in file_list and each row corresponds to a pixel.

    Notes:
    The function assumes that all files have the same dimensions and can be loaded using np.loadtxt.
    """
   
    file_path = direc+file_list[0]                     # Use first image to get the number of pixels
    array = np.loadtxt(file_path)
    ny,nx = array.shape
    print("nx, ny ",nx,ny)
    num_pixels = nx*ny 
    
    A = np.zeros((num_pixels,len(file_list))) 
    A[:,0] = np.reshape(array,(num_pixels,))
    ts = time.time()
    for i,f in enumerate(file_list[1:]):
        file_path = direc+f 
        array = np.loadtxt(file_path)
        A[:,i] = np.reshape(array,(num_pixels,))
    if save_file == True:
        #file_path = out_direc + "A288x384_july2025.npy"
        np.save(out_path,A)
    print(" Finished getting A ",time.time() - ts)
   
    
    return A        




def main():

    print(" main ")
    

if __name__ == "__main__":
    main()

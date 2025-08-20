
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

import os 


def get_3slit_mask(H,W):

    mask = np.zeros((H,W))
    print(" mask ",mask.shape)
    num_mask_pixels = W*H
    xc , yc = W//2, H//2
    ws = 6 # H//6 #2*6               # width  slit
    wsd2 = ws//2
    hs = 2*H//3 # 2*10           # height slit
    hsd2 = hs//2
    mask[xc-hsd2:xc+hsd2,yc-wsd2:yc+wsd2] = 1
    yc1 =  H//4 #9              # offset of left right slit
    yc2 =  -H//4 #-9
    mask[xc-hsd2:xc+hsd2,yc+yc1-wsd2:yc+yc1+wsd2] = 1
    mask[xc-hsd2:xc+hsd2,yc+yc2-wsd2:yc+yc2+wsd2] = 1
    #plt.imshow(mask)
    #plt.figure()
    return mask

def get_mask(H,W,mask_type="3-slit"):

     if mask_type == "3-slit":
         return get_3slit_mask(H,W)
         
def main():

    print("main")
    
if __name__ == "__main__":
    main()

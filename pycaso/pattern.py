#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
import numpy as np
import cv2

def ChAruco_board (ncx : int,
                   ncy : int,
                   pixel_factor : int = 1) -> np.ndarray :
    """Create the image for ChAruco board
    
    Args:
       ncx : int
           number of cases along x axis
       ncy : int
           number of cases along y axis
       pixel_factor : int
           Define the number of pixel per ChAruco pixel
           
    Returns:
       ChAruco_Board_build_img : np.ndarray
           ChAruco image
    """
    n = ncx * ncy / 2
    ChAruco_Board_img = []
    for e in range (int(n)) :
        if cv2.__version__>='4.7.0' :
            dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
            ChAruco_mrk = cv2.aruco.generateImageMarker(dictionary, e, pixel_factor*8)
        else :
            dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_1000)
            ChAruco_mrk = cv2.aruco.drawMarker(dictionary, e, pixel_factor*8)
        x, y = ChAruco_mrk.shape
        dx, dy = x//2, y//2
        black_square = np.ones ((x//2,y//2)) * 255
        ChAruco_square = np.empty((2*x, 2*y))
        for i in range (4) :
            for j in range (4) :
                if (i == 0) or (i == 3) or (j == 0) or (j == 3) :
                    ChAruco_square[dx*i:dx*(i+1),dy*j:dy*(j+1)] = black_square
                else :
                    ChAruco_square[dx*i:dx*(i+1),dy*j:dy*(j+1)] = ChAruco_mrk[dx*(i-1):dx*i,dy*(j-1):dy*j]          
        ChAruco_Board_img.append(ChAruco_square)
    x, y = ChAruco_Board_img[0].shape
    white_square = np.zeros ((x,y))
    shapetot = (y*ncy, x*ncx)
    ChAruco_Board_build_img = np.empty(shapetot)
    e = 0
    for i in range(ncy) :
        for j in range(ncx) :
            if ((i+j)%2) == 0 :
                ChAruco_Board_build_img[x*i:x*(i+1),y*j:y*(j+1)] = ChAruco_Board_img[e]
                e += 1
            else :
                ChAruco_Board_build_img[x*i:x*(i+1),y*j:y*(j+1)] = white_square
    
    fig = plt.imshow(ChAruco_Board_build_img, interpolation='nearest', cmap='gray', vmin=0, vmax=255)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.imsave('ChAruco_Board_build_img.png', ChAruco_Board_build_img, cmap = 'gray')
    plt.show()
    return (ChAruco_Board_build_img)

if __name__ == '__main__' :
    # Show the reals and theoreticals points  
    ncx = 16
    ncy = 12
    pixel_factor = 10 
    Pattern = ChAruco_board(ncx,
                            ncy,
                            pixel_factor)
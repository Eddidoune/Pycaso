#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 09:19:07 2021

@author: caroneddy
"""
#import argparse
import numpy as np
import sys
import os
sys.path.append("/home/caroneddy/These/Stereo_camera/Socapy/Stereo_libraries")
from matplotlib import pyplot as plt

import soloff_library as soloff
import data_library as data
import math

from glob import glob

def magnification (X1, X2, x1, x2) :
    """Calculation of the magnification between reals and detected positions
    
    Args:
       X1 : numpy.ndarray
           Organised real positions (X1 = X axe)
       X2 : numpy.ndarray
           Organised real positions (X2 = Y axe)
       x1 : numpy.ndarray
           Organised detected positions (x1 = x axe)
       x1 : numpy.ndarray
           Organised detected positions (x2 = y axe)
    Returns:
       Magnification : int
           Magnification between reals and detected positions
    """
    Delta_X1 = np.mean(abs(X1-np.mean(X1)))
    Delta_X2 = np.mean(abs(X2-np.mean(X2)))
    Delta_x1 = np.mean(abs(x1-np.mean(x1)))
    Delta_x2 = np.mean(abs(x2-np.mean(x2)))
    Magnification = np.asarray([Delta_x1/Delta_X1, Delta_x2/Delta_X2]) 
    return (Magnification)


if __name__ == '__main__' :
    __calibration_dict__ = {
    'left_calibration_folder' : './Images_example/left-calib',
    'right_calibration_folder' : './Images_example/right-calib',
    'name' : 'micro_calibration',
    'ncx' : 16,
    'ncy' : 12,
    'sqr' : 0.3}  #in mm
    
    x3_list = np.asarray([-0.005, -0.010, -0.015, -0.020, -0.025, -0.030, -0.035, -0.040, 0.000, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040])
    saving_folder = 'TXT_example'
    
    __identification_dict__ = {
    'left_calibration_folder' : './Images_example/left-identification',
    'right_calibration_folder' : './Images_example/right-identification',
    'name' : 'micro_identification',
    'ncx' : 16,
    'ncy' : 12,
    'sqr' : 0.3}  #in mm
         
    # Create the result folder if not exist
    if os.path.exists(saving_folder) :
        ()
    else :
        os.makedirs(saving_folder)

    print('')
    print('#####       ')
    print('Start calibration')
    print('#####       ')
    # Detect points from folders
    all_Ucam, all_Xref = data.detection(__calibration_dict__,
                                   detection = False,
                                   saving_folder = saving_folder)        

    # Creation of the reference matrix Xref and the real position Ucam for each camera i
    xc1, xc2, Xc1, Xc2 = data.camera_np_coordinates(all_Ucam, 
                                               all_Xref, 
                                               x3_list)
    
    # Plot the references plans
    soloff.refplans(xc1, x3_list)
    
    
    # Calcul of the Soloff polynome's constants. X = A . M
    A111 = np.zeros((2, 2, 4))
    A221 = np.zeros((2, 2, 9))
    A222 = np.zeros((2, 2, 10))
    A332 = np.zeros((2, 2, 19))
    A333 = np.zeros((2, 2, 20))
    A443 = np.zeros((2, 2, 34))
    A444 = np.zeros((2, 2, 35))
    A554 = np.zeros((2, 2, 55))
    A555 = np.zeros((2, 2, 56))
    polynome_degrees = [111,221,222,332,333,443,444,554,555]
    AAA = [A111,A221,A222,A332,A333,A443,A444,A554,A555]
    # polynome_degrees = [111,221,222,332]
    # AAA = [A111,A221,A222,A332]
    Magnification = np.zeros((2, 2))
    for i in [1, 2] :
        if i == 1 :
            x, X = xc1, Xc1
        elif i == 2 :
            x, X = xc2, Xc2
        x1, x2, x3 = x
        X1, X2 = X
        
        # Compute the magnification (same for each cam as set up is symetric)
        Magnification[i-1] = magnification (X1, X2, x1, x2)
        
        for p in range (len (AAA)) :
            # Do the system X = Ai*M, where M is the monomial of the real coordinates of crosses and X the image coordinates, and M the unknow (polynomial form aab)
            polynome_degree = polynome_degrees[p]
            M = soloff.Polynome({'polynomial_form' : polynome_degree}).pol_form(x)
            Ai = np.matmul(X, np.linalg.pinv(M))
            AAA[p][i-1] = Ai
    
            # Error of projection
            Xd = np.matmul(Ai,M)
            proj_error = X - Xd
            print('Max ; min projection error (polynomial form ' + str(polynome_degree) + ') for camera ' + str(i) + ' = ' + str(np.amax(proj_error))+ ' ; ' + str(np.amin(proj_error)) + ' px')

    print('')
    print('#####       ')
    print('End calibration')
    print('#####       ')
    print('')

    all_Ucam_more, all_Xref_more = data.detection(__identification_dict__,
                                   detection = False,
                                   saving_folder = saving_folder)
    Images_left = sorted(glob(__identification_dict__['left_calibration_folder'] + '/*.tif'))
    Nimg = len(all_Ucam_more)//2
    folder =__identification_dict__['left_calibration_folder']
    for i in range (1) :
        Xc1_identification, Xc2_identification = all_Ucam_more[i], all_Ucam_more[i+Nimg]
    
        # We're searching for the solution x0(x1, x2, x3) as Xc1 = ac1 . (1 x1 x2 x3) and Xc2 = ac2 . (1 x1 x2 x3)  using least square method.
        x0 = soloff.least_square_method (Xc1_identification, Xc2_identification, A111)
        
        # Solve the polynomials constants ai with curve-fit method (Levenberg Marcquardt - 332)
        title = 'Reconstruction ( curve_fit method ; polynomial_form : 332) ; ' + Images_left[i][len(folder)+1:-3]
        xopt1, Xcalculated, Xdetected = soloff.Levenberg_Marquardt_solving(Xc1_identification, Xc2_identification, A332, x0, polynomial_form = 332, method = 'curve_fit', title = title)
        
        # # Solve the polynomials constants ai with curve-fit method (Levenberg Marcquardt - 333)
        title = 'Reconstruction ( curve_fit method ; polynomial_form : 443) ; ' + Images_left[i][len(folder)+1:-3]
        xopt1, Xcalculated, Xdetected = soloff.Levenberg_Marquardt_solving(Xc1_identification, Xc2_identification, A443, x0, polynomial_form = 443, method = 'curve_fit', title = title) 

    sys.exit()



    print('')
    print('#####       ')
    print('Direct method')
    print('#####       ')
    print('')

    direct_AAA1 = np.zeros((3, 5))
    direct_AAA2 = np.zeros((3, 15))
    direct_AAA3 = np.zeros((3, 35))
    direct_AAA4 = np.zeros((3, 70))
    direct_AAA = [direct_AAA1, direct_AAA2, direct_AAA3, direct_AAA4]
    direct_polynome_degrees = [1,2,3,4]
    for p in range (len (direct_AAA)) :
        # Do the system x = Ap*M, where M is the monomial of the real coordinates of crosses and x the image coordinates, and M the unknow
        x = xc1
        polynome_degree = direct_polynome_degrees[p]
        M = soloff.Direct_Polynome({'polynomial_form' : polynome_degree}).pol_form(Xc1, Xc2)
        Ap = np.matmul(x, np.linalg.pinv(M))
        direct_AAA[p] = Ap

        # Error of projection
        xd = np.matmul(Ap,M)
        proj_error = x - xd
        print('Max ; min projection error (polynomial form ' + str(polynome_degree) + ') for camera ' + str(i) + ' = ' + str(np.amax(proj_error))+ ' ; ' + str(np.amin(proj_error)) + ' px')

    for i in range (Nimg) :
        imgc1, imgc2 = all_Ucam_more[i], all_Ucam_more[i+Nimg]
        
        # Solve by direct method
        Xl1, Xl2 = imgc1[:,0], imgc1[:,1]
        Xr1, Xr2 = imgc2[:,0], imgc2[:,1]
        Xl = np.zeros((2,len(Xl1)))
        Xr = np.zeros((2,len(Xr1)))
        Xl = Xl1, Xl2
        Xr = Xr1, Xr2
        
        M = soloff.Direct_Polynome({'polynomial_form' : 4}).pol_form(Xl, Xr)
        xopt = np.matmul(direct_AAA[3],M)
        soloff.fit_plans_to_points(xopt.reshape((1,xopt.shape[0], xopt.shape[1])), 
                                    title = 'Test direct',
                                    axes_xyz = 1,
                                    label = 'xs')



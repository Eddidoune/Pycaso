#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 09:19:07 2021

@author: caroneddy
"""
#import argparse
import numpy as np
import scipy.optimize as sopt
import sys
sys.path.append("/home/caroneddy/These/Stereo_camera/Stereo_libraries")
from matplotlib import pyplot as plt

import soloff_library as soloff
import data_library as data
import math

from glob import glob



def detection (__dict__,
               detection = True,
               saving_folder = 'Folders_npy') :
    """Detect the corners of Charucco's pattern.
    
    Args:
       camera_side : str
           'left' or 'right'
       __dict__ : dict
           Pattern properties define in a dict.
       detection : bool, optional
           If True, all the analysis will be done. If False, the code will take the informations in 'saving_folder'
       saving_folder : str, optional
           Where to save datas
    Returns:
       all_Ucam : numpy.ndarray
           The corners of the pattern detect by the camera
       all_Xref : numpy.ndarray
           The theorical corners of the pattern
    """
    # Taking the main parameters from bibliotheque_data_eddy.
    left_calibration_folder = __dict__['left_calibration_folder']
    right_calibration_folder = __dict__['right_calibration_folder']
    name = __dict__['name']
    ncx = __dict__['ncx']
    ncy = __dict__['ncy']
    sqr = __dict__['sqr']
    Images_left = sorted(glob(str(left_calibration_folder) + '/*.tif'))
    Images_right = sorted(glob(str(right_calibration_folder) + '/*.tif'))
    Images = Images_left
    for i in range (len(Images_right)) :
        Images.append(Images_right[i])
    
    Save_Ucam_Xref = [str(saving_folder) +"/all_Ucam_" + name + ".npy", str(saving_folder) + "/all_Xref_" + name + ".npy"]
    
    # Corners detection
    if detection :
        print('    - Detection of the pattern in progress ...')
        # Creation of the theoretical pattern + detection of camera's pattern
        Xref = data.Modele_calibration(ncx, ncy, sqr)
        all_Xref, all_Ucam = data.Modeles_calibration_images_tronquage(Images, Xref, __dict__)
        
        all_Xref = np.asarray(all_Xref)
        all_Xref = all_Xref[:, :, [0, 1]]
        all_Ucam = np.asarray(all_Ucam)
        all_Ucam = all_Ucam[:, :, [0, 1]]

        np.save(Save_Ucam_Xref[0], all_Ucam)
        np.save(Save_Ucam_Xref[1], all_Xref)
        
        print('    - Saving datas in ', saving_folder)
    # Taking pre-calculated datas from the saving_folder
    else :
        print('    - Taking datas from ', saving_folder)        
        all_Ucam = np.load(Save_Ucam_Xref[0])
        all_Xref = np.load(Save_Ucam_Xref[1])
    return(all_Ucam, all_Xref)

def camera_np_coordinates (all_Ucam, 
                           all_Xref, 
                           x3_list) :
    """Organising the coordinates of the calibration
    
    Args:
       all_Ucam : numpy.ndarray
           The corners of the pattern detect by the camera
       all_Xref : numpy.ndarray
           The theorical corners of the pattern
       x3_list : numpy.ndarray
           List of the different z position. (Ordered the same way in the target folder)
       saving_folder : str, optional
           Where to save datas
    Returns:
       xc1 : numpy.ndarray
           Organised real positions of camera 1
       xc2 : numpy.ndarray
           Organised real positions of camera 2
       Xc1 : numpy.ndarray
           Organised detected positions of camera 1
       Xc2 : numpy.ndarray
           Organised detected positions of camera 2
    """
    for i in [1, 2] :
        print('')
        mid = all_Ucam.shape[0]//2    
        all_Ucami = all_Ucam[(i-1)*mid:i*mid,:,:]
        all_Xrefi = all_Xref[i*(mid-1):i*mid,:,:]
        sU = all_Ucami.shape
        Xref = all_Xrefi[0]
        all_Xrefi = np.empty ((sU[0], sU[1], sU[2]+1))
        x = np.empty ((sU[0] * sU[1], sU[2]+1))
        X = np.empty ((sU[0] * sU[1], sU[2]))
        for j in range (sU[0]) :
            all_Xrefi[j][:,0] = Xref[:,0]
            all_Xrefi[j][:,1] = Xref[:,1]
            all_Xrefi[j][:,2] = x3_list[j]

            x[j*sU[1] : (j+1)*sU[1], :]  = all_Xrefi[j]
            X[j*sU[1] : (j+1)*sU[1], :]  = all_Ucami[j]

        # Real position in space : Xref (x1, x2, x3)
        x1 = x[:,0]
        x2 = x[:,1]
        x3 = x[:,2]
        x = np.asarray([x1,x2,x3]) # reshape x

        # Position detected from cameras : Ucam (X1, X2)
        X1 = X[:,0]
        X2 = X[:,1]
        X = np.asarray([X1,X2]) # reshape X
        if i == 1 :
            xc1, Xc1 = x, X
        if i == 2 :
            xc2, Xc2 = x, X
        # Plot the plans x3 = -10 ; -5 ; 0 ; 5 ; 10 and the real positions x
        # soloff.fit_plans_to_points(all_Xrefi, title = 'Calibration ; camera ' + str(i))
    return (xc1, xc2, Xc1, Xc2)

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

def least_square_method (Xc1, 
                         Xc2, 
                         A111) :
    """Resolve by least square method the system A . x = X for each points detected and both cameras
    
    Args:
       Xc1 : numpy.ndarray
           Organised real positions of camera 1
       Xc2 : numpy.ndarray
           Organised real positions of camera 2
       A111 : numpy.ndarray
           Constants of the first order calibration polynome
    Returns:
       x0 : numpy.ndarray
           Solution x = xsol of the system 
    """
    N = len (Xc1)
    x0 = np.zeros((3, N))
    for i in range (N) :
        X1c1, X2c1 = Xc1[i,0], Xc1[i,1]
        X1c2, X2c2 = Xc2[i,0], Xc2[i,1]
        a1c1, a2c1 = A111[0,0,:], A111[0,1,:]
        a1c2, a2c2 = A111[1,0,:], A111[1,1,:]
    
        A = np.array([a1c1, a2c1, a1c2, a2c2])
        X = np.array([X1c1-a1c1[0], X2c1-a2c1[0], X1c2-a1c2[0], X2c2-a2c2[0]])
        
        A = A[:,1:4]
        At = np.transpose (A)
        J = np.matmul(At, A)
        J_ = np.linalg.inv(J)
        XA = np.matmul(X, A)
        
        x0[:, i] = np.matmul(J_, XA)
    
    return (x0)    

def Levenberg_Marquardt_solving (imgc1, 
                                 imgc2, 
                                 A, 
                                 x0, 
                                 polynomial_form, 
                                 method = 'curve_fit', 
                                 img = False,
                                 folder = '') :
    N = len(x0[0])    
    Xdetected = np.array([imgc1[:,0], imgc1[:,1], imgc2[:,0], imgc2[:,1]])
    X0 = Xdetected.reshape((4*N))
    A0 = np.array([A[0,0], A[0,1], A[1,0], A[1,1]])
    method
    x0 = x0.reshape((3*N))
    if method == 'curve_fit' :
        xopt, pcov = sopt.curve_fit(soloff.Polynome({'polynomial_form' : polynomial_form}).polynomial_LM, 
                                    A0, 
                                    X0, 
                                    p0 = x0, 
                                    method ='lm')

    elif method == 'least_squares' :
        resultat = sopt.least_squares(soloff.Polynome({'polynomial_form' : polynomial_form}).polynomial_Least_Square, 
                                  x0, 
                                  args = (X0,A0))  
        xopt = resultat.x
    
    xopt = np.array(xopt)
    xopt = xopt.reshape((3,N))
    Yopt = soloff.Polynome({'polynomial_form' : polynomial_form}).polynomial_system(xopt, A0)
    Xdiff = Yopt - Xdetected
    print(str(polynomial_form), ' : The max error between detected and calculated points is ', np.max(Xdiff), ' pixels.')
    
    # plt.plot(x00, y00, z00, 'r.', label = 'x0')
    L_file_ext = len(folder)+1
    if img != False :
        title = str(polynomial_form) + ' : Reconstruction (with ' + method + ' method) ; ' + img[i][L_file_ext:-3]
    else : 
        title = str(polynomial_form) + ' : Reconstruction (with ' + method + ' method)'
    soloff.fit_plans_to_points(xopt.reshape((1,xopt.shape[0], xopt.shape[1])), 
                                title = title,
                                axes_xyz = 1,
                                label = 'xs')
    return (xopt, Yopt, Xdetected)

if __name__ == '__main__' :
    __calibration_dict__ = {
    'left_calibration_folder' : './Images/micro/left-test',
    'right_calibration_folder' : './Images/micro/right-test',
    'name' : 'micro_calibration',
    'ncx' : 16,
    'ncy' : 12 ,
    'sqr' : 300*(10**(-3))}  #in mm
    
    x3_list = np.asarray([-0.005, -0.010, -0.015, -0.020, -0.025, -0.030, -0.035, -0.040, 0.000, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040])
    saving_folder = 'Fichiers_txt'
    
    __identification_dict__ = {
    'left_calibration_folder' : './Images/micro/left-test-more',
    'right_calibration_folder' : './Images/micro/right-test-more',
    'name' : 'micro_identification',
    'ncx' : 16,
    'ncy' : 12 ,
    'sqr' : 300*(10**(-3))}  #in mm
         
    
    print('')
    print('#####       ')
    print('Start calibration')
    print('#####       ')

    # Detect points from folders
    all_Ucam, all_Xref = detection(__calibration_dict__,
                                   detection = False,
                                   saving_folder = saving_folder)        

    # Creation of the reference matrix Xref and the real position Ucam for each camera i
    xc1, xc2, Xc1, Xc2 = camera_np_coordinates(all_Ucam, 
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
    polynome_degrees = [111,221,222,332]
    AAA = [A111,A221,A222,A332]
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

    all_Ucam_more, all_Xref_more = detection(__identification_dict__,
                                   detection = False,
                                   saving_folder = saving_folder)
    Images_left = sorted(glob(__identification_dict__['left_calibration_folder'] + '/*.tif'))
    Nimg = len(all_Ucam_more)//2
    for i in range (1) :
        imgc1, imgc2 = all_Ucam_more[i], all_Ucam_more[i+Nimg]
    
        # We're searching for the solution x0(x1, x2, x3) as Xc1 = ac1 . (1 x1 x2 x3) and Xc2 = ac2 . (1 x1 x2 x3)  using least square method.
        x0 = least_square_method (imgc1, imgc2, A111)
        
        # Solve the polynomials constants ai with curve-fit method (Levenberg Marcquardt - 332)
        xopt1, Yopt, Xdetected = Levenberg_Marquardt_solving(imgc1, imgc2, A332, x0, polynomial_form = 332, method = 'least_squares', img = Images_left, folder = __identification_dict__['left_calibration_folder'])     
        # xopt1, Yopt, Xdetected = Levenberg_Marquardt_solving(imgc1, imgc2, A332, x0, polynomial_form = 332, method = 'curve_fit', img = Images_left, folder = __identification_dict__['left_calibration_folder'])
        
        # # Solve the polynomials constants ai with curve-fit method (Levenberg Marcquardt - 333)
        # xopt2 = Levenberg_Marquardt_solving(imgc1, imgc2, A333, x0, polynomial_form = 333, method = 'curve_fit', img = Images_left) 



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



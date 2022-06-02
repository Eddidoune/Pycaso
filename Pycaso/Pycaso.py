#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 09:19:07 2021

@author: Eddidoune
"""
import sigfig as sgf
try : 
    import cupy as np
except ImportError:
    import numpy as np
import sys
import pathlib
import os
import time
from glob import glob
import pandas as pd
import cv2
import solve_library as solvel 
import data_library as data
import matplotlib.pyplot as plt
import scipy.ndimage as nd
# from mpl_toolkits import mplot3d
import csv
import math

def magnification (X1, X2, x1, x2) :
    """Calculation of the magnification between reals and detected positions
    
    Args:
       X1 : numpy.ndarrayx
           Organised detected positions (X1 = X axe)
       X2 : numpy.ndarray
           Organised detected positions (X2 = Y axe)
       x1 : numpy.ndarray
           Organised real positions (x1 = x axe)
       x2 : numpy.ndarray
           Organised real positions (x2 = y axe)
    Returns:
       Magnification : int
           Magnification between detected and real positions
           [Mag x, Mag y]
    """
    Delta_X1 = np.nanmean(abs(X1-np.nanmean(X1)))
    Delta_X2 = np.nanmean(abs(X2-np.nanmean(X2)))
    Delta_x1 = np.nanmean(abs(x1-np.nanmean(x1)))
    Delta_x2 = np.nanmean(abs(x2-np.nanmean(x2)))
    Magnification = np.asarray([Delta_x1/Delta_X1, Delta_x2/Delta_X2]) 
    return (Magnification)

def Soloff_calibration (__calibration_dict__,
                        x3_list,
                        Soloff_pform,
                        hybrid_verification = False) :
    """Calculation of the magnification between reals and detected positions 
    and the calibration parameters A = A111 (Resp A_pol):--> X = A.M(x)
    
    Args:
       __calibration_dict__ : dict
           Calibration properties define in a dict. Including 'left_folder', 
           'right_folder', 'name', 'ncx', 'ncy', 'sqr'
       x3_list : numpy.ndarray
           List of the different z position. (WARNING : Should be order the 
                                              same way in the target folder)
       Soloff_pform : int
           Polynomial form
       hybrid_verification : bool, optional
           If True, verify each pattern detection and propose to pick 
           manually the bad detected corners. The image with all detected
           corners is show and you can decide to change any point using
           it ID (ID indicated on the image) as an input. If there is no
           bad detected corner, press ENTER to go to the next image.

    Returns:
       A111 : numpy.ndarray
           Constants of Soloff polynomial form '111'
       A_pol : numpy.ndarray
           Constants of Soloff polynomial form chose (Soloff_pform)
       Magnification : int
           Magnification between reals and detected positions 
           [[Mag Left x, Mag Left y], [Mag Right x, Mag Right y]]
    """
    
    A111 = np.zeros((2, 2, 4))
    if Soloff_pform == 111 :
        A_pol = np.zeros((2, 2, 4))
    elif Soloff_pform == 221 :
        A_pol = np.zeros((2, 2, 9))
    elif Soloff_pform == 222 :
        A_pol = np.zeros((2, 2, 10))
    elif Soloff_pform == 332 :
        A_pol = np.zeros((2, 2, 19))
    elif Soloff_pform == 333 :
        A_pol = np.zeros((2, 2, 20))
    elif Soloff_pform == 443 :
        A_pol = np.zeros((2, 2, 34))
    elif Soloff_pform == 444 :
        A_pol = np.zeros((2, 2, 35))
    elif Soloff_pform == 554 :
        A_pol = np.zeros((2, 2, 55))
    elif Soloff_pform == 555 :
        A_pol = np.zeros((2, 2, 56))    
    else :
        print ('Only define for polynomial forms 111, 221, 222, 332, 333, 443, 444, 554 or 555')
        sys.exit()
    
    A_0 = [A111, A_pol]
    Soloff_pforms = [111, Soloff_pform]

    
    # Detect points from folders
    all_Ucam, all_Xref, nb_pts = data.pattern_detection(__calibration_dict__,
                                                        hybrid_verification = hybrid_verification)        

    # Creation of the reference matrix Xref and the real position Ucam for 
    # each camera
    x, Xc1, Xc2 = data.camera_np_coordinates(all_Ucam, all_Xref, x3_list)     

    # Plot the references plans
    solvel.refplans(x, x3_list)

    # Calcul of the Soloff polynome's constants. X = A . M
    Magnification = np.zeros((2, 2))
    for camera in [1, 2] :
        if camera == 1 :
            X = Xc1
        elif camera == 2 :
            X = Xc2
        x1, x2, x3 = x
        X1, X2 = X
        
        # Compute the magnification (same for each cam as set up is symetric)
        Magnification[camera-1] = magnification (X1, X2, x1, x2)
        
        for pol in range (len (A_0)) :
            # Do the system X = Ai*M, where M is the monomial of the real 
            # coordinates of crosses and X the image coordinates, and M the 
            # unknow (polynomial form aab)
            Soloff_pform = Soloff_pforms[pol]
            M = solvel.Soloff_Polynome({'polynomial_form' : Soloff_pform}).pol_form(x)
            Ai = np.matmul(X, np.linalg.pinv(M))
            A_0[pol][camera-1] = Ai
    
            # Error of projection
            Xd = np.matmul(Ai,M)
            proj_error = X - Xd
            print('Max ; min projection error (polynomial form ', 
                  str(Soloff_pform),
                  ') for camera ', 
                  str(camera),
                  ' = ',
                  str(sgf.round(np.nanmax(proj_error), sigfigs =3)),
                  ' ; ',
                  str(sgf.round(np.nanmin(proj_error), sigfigs =3)),
                  ' px')
    A111, A_pol = A_0
    return(A111, A_pol, Magnification)

def Soloff_identification (Xc1_identified,
                           Xc2_identified,
                           A111, 
                           A_pol,
                           Soloff_pform,
                           method = 'curve_fit') :
    """Identification of the points detected on both cameras left and right 
    into the global 3D-space
    
    Args:
       Xc1_identified : numpy.ndarray
           Points identified on the left camera
       Xc2_identified : numpy.ndarray
           Points identified on the right camera
       A111 : numpy.ndarray
           Constants of Soloff polynomial form '111'
       A_pol : numpy.ndarray
           Constants of Soloff polynomial form chose (polynomial_form)
       Soloff_pform : int
           Polynomial form
       method : str, optional
           Python method used to solve it ('Least-squares' or 'curve-fit')

    Returns:
       x_solution : numpy.ndarray
           Identification in the 3D space of the detected points
    """
    
    # We're searching for the solution x0(x1, x2, x3) as Xc1 = ac1 . 
    # (1 x1 x2 x3) and Xc2 = ac2 . (1 x1 x2 x3)  using least square method.
    x0 = solvel.least_square_method (Xc1_identified, Xc2_identified, A111)
    
    # Solve the polynomials constants ai with curve-fit method (Levenberg 
    # Marcquardt)
    x_solution, Xc, Xd = solvel.Levenberg_Marquardt_solving(Xc1_identified, 
                                                            Xc2_identified, 
                                                            A_pol, 
                                                            x0, 
                                                            Soloff_pform = Soloff_pform, 
                                                            method = 'curve_fit')
    return (x_solution)


def direct_calibration (__calibration_dict__,
                        x3_list,
                        direct_pform) :
    """Calculation of the magnification between reals and detected positions 
    and the calibration parameters A:--> x = A.M(X)
    
    Args:
       __calibration_dict__ : dict
           Calibration properties define in a dict. Including 'left_folder', 
           'right_folder', 'name', 'ncx', 'ncy', 'sqr'
       x3_list : numpy.ndarray
           List of the different z position. (WARNING : Should be order the 
                                              same way in the target folder)
       direct_pform : int
           Polynomial degree

    Returns:
       A : numpy.ndarray
           Constants of direct polynomial
       Magnification : int
           Magnification between reals and detected positions
    """
    
    if direct_pform == 1 :
        direct_A = np.zeros((3, 5))
    elif direct_pform == 2 :
        direct_A = np.zeros((3, 15))
    elif direct_pform == 3 :
        direct_A = np.zeros((3, 35))
    elif direct_pform == 4 :
        direct_A = np.zeros((3, 70))
    else :
        print ('Only define for polynomial degrees (1, 2, 3 or 4')
        sys.exit()
    # Detect points from folders
    all_Ucam, all_Xref, nb_pts = data.pattern_detection(__calibration_dict__)        

    # Creation of the reference matrix Xref and the real position Ucam for 
    # each camera i
    x, Xc1, Xc2 = data.camera_np_coordinates(all_Ucam, all_Xref, x3_list)

    # Plot the references plans
    solvel.refplans(x, x3_list)

    # Calcul of the Soloff polynome's constants. X = A . M
    Magnification = np.zeros((2, 2))

    for camera in [1, 2] :
        if camera == 1 :
            X = Xc1
        elif camera == 2 :
            X = Xc2
        x1, x2, x3 = x
        X1, X2 = X
        
        # Compute the magnification (same for each cam as set up is symetric)
        Magnification[camera-1] = magnification (X1, X2, x1, x2)
        
        # Do the system x = Ap*M, where M is the monomial of the real 
        # coordinates of crosses and x the image coordinates, and M the unknow
        M = solvel.Direct_Polynome({'polynomial_form' : direct_pform}).pol_form(Xc1, Xc2)
        Ap = np.matmul(x, np.linalg.pinv(M))
        direct_A = Ap

        # Error of projection
        xd = np.matmul(Ap,M)
        proj_error = x - xd
        print('Max ; min projection error (polynomial form ',
              str(direct_pform),
              ') for camera ',
              str(camera),
              ' = ',
              str(sgf.round(np.amax(proj_error), sigfigs = 3)),
              ' ; ',
              str(sgf.round(np.amin(proj_error), sigfigs = 3)),
              ' px')
    return(direct_A, Magnification)    

def direct_identification (Xc1_identified,
                           Xc2_identified,
                           direct_A,
                           direct_pform) :
    """Identification of the points detected on both cameras left and right 
    into the global 3D-space
    
    Args:
       Xc1_identified : numpy.ndarray
           Points identified on the left camera
       Xc2_identified : numpy.ndarray
           Points identified on the right camera
       direct_A : numpy.ndarray
           Constants of direct polynomial
       direct_pform : int
           Polynomial form


    Returns:
       x_solution : numpy.ndarray
           Identification in the 3D space of the detected points
    """    
    # Solve by direct method
    Xl1, Xl2 = Xc1_identified[:,0], Xc1_identified[:,1]
    Xr1, Xr2 = Xc2_identified[:,0], Xc2_identified[:,1]
    Xl = np.zeros((2,len(Xl1)))
    Xr = np.zeros((2,len(Xr1)))
    Xl = Xl1, Xl2
    Xr = Xr1, Xr2
    
    M = solvel.Direct_Polynome({'polynomial_form' : direct_pform}).pol_form(Xl, Xr)
    xsolution = np.matmul(direct_A,M)
    return(xsolution)

def AI_training (X_c1,
                 X_c2,
                 xSoloff_solution,
                 AI_training_size = 1000,
                 file = 'Soloff_AI_training.csv') :
    """Training the AI metamodel with already known datas.
    
    Args:
       X_c1 : numpy.ndarray
           Left coordinates of points.
       X_c2 : numpy.ndarray
           Right coordinates of points.
       xSoloff_solution : numpy.ndarray
           3D space coordinates identified with Soloff method.
       AI_training_size : int
           Number of datas (points) used to train the AI metamodel.
       file : str
           Name of saving file for training

    Returns:
       model : sklearn.ensemble._forest.RandomForestRegressor
           AI model
    """
    # Organise the list of parameters
    Xl0, Yl0 = X_c1[:,0], X_c1[:,1]
    Xr0, Yr0 = X_c2[:,0], X_c2[:,1]
    xS, yS, zS = xSoloff_solution
    N = X_c1.shape[0]
    if AI_training_size > N :
        print('AI_training_size reduced to size ', N)
        AI_training_size = N
    else :
        ()
    
    # List of random points (To minimize the size of calculation)
    rd_list = np.ndarray.astype((np.random.rand(AI_training_size)*N),int)    
    with open(file, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, 
                                delimiter=' ',
                                quotechar='|', 
                                quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['Xl'] + ['Yl'] + ['Xr'] + ['Yr'] + 
                            ['x'] + ['y'] + ['z'])
        for i in range (AI_training_size) :
            spamwriter.writerow([str(Xl0[rd_list[i]]), 
                                 str(Yl0[rd_list[i]]), 
                                 str(Xr0[rd_list[i]]), 
                                 str(Yr0[rd_list[i]]), 
                                 str(xS[rd_list[i]]), 
                                 str(yS[rd_list[i]]), 
                                 str(zS[rd_list[i]])])
    # Build the AI model with the AI_training_size.
    model, accuracy = solvel.AI_solve (file)
    return(model)

def AI_identification (X_c1,
                       X_c2,
                       model) :
    """Calculation of the 3D points with AI model.
    
    Args:
       X_c1 : numpy.ndarray
           Left coordinates of points.
       X_c2 : numpy.ndarray
           Right coordinates of points.
       model : sklearn.ensemble._forest.RandomForestRegressor
           AI model

    Returns:
       xAI_solution : numpy.ndarray
           3D space coordinates identified with AI method.
    """
    # Solve on all pts
    X = np.transpose(np.array([X_c1[:,0], X_c1[:,1], 
                               X_c2[:,0], X_c2[:,1]]))        
    xAI_solution = model.predict(X)
    xAI_solution = np.transpose(xAI_solution)
    return (xAI_solution)


if __name__ == '__main__' :
    main_path = "/home/caroneddy/These/Stereo_camera/Pycaso_archives/src"    
    saving_folder = main_path + '/results/2022_04_15_results_adrien_micromachine/501_sample'
    
    # Define the inputs
    __calibration_dict__ = {
    'left_folder' : main_path + '/Images_example/2022_04_15_Adrien/left_501_x2',
    'right_folder' : main_path + '/Images_example/2022_04_15_Adrien/right_501_x2',
    'name' : 'micro_calibration',
    'saving_folder' : saving_folder,
    'ncx' : 16,
    'ncy' : 12,
    'sqr' : 0.3}  #in mm
    
    # __pattern_dict__ = {
    # 'left_folder' : main_path + '/Images_example/2022_04_15_Adrien/left_sample_identification',
    # 'right_folder' : main_path + '/Images_example/2022_04_15_Adrien/right_sample_identification',
    # 'name' : 'micro_identification',
    # 'ncx' : 12,
    # 'ncy' : 12,
    # 'sqr' : 0.3}  #in mm
    
    __DIC_dict__ = {
    'left_folder' : main_path + '/Images_example/2022_04_15_Adrien/left_sample_identification',
    'right_folder' : main_path + '/Images_example/2022_04_15_Adrien/right_sample_identification',
    'name' : 'micro_identification',
    'saving_folder' : saving_folder,
    'window' : [[200, 1800], [200, 1800]]}  #in mm
    
    profilo = '/home/caroneddy/These/Stereo_camera/Pycaso_archives/src/results/2022_04_15_results_adrien_micromachine/profilo/x5_stitching5x5_tilt_cyl_removed_overlap40.npy'
    
    
    # Create the list of z plans
    Folder = __calibration_dict__['left_folder']
    Imgs = sorted(glob(str(Folder) + '/*'))
    x3_list = np.zeros((len(Imgs)))
    for i in range (len(Imgs)) :
        x3_list[i] = float(Imgs[i][len(Folder)+ 1:-4])
    # x3_list = np.array(sorted(x3_list))

    
    # Chose the polynomial degree for the calibration fitting
    Soloff_pform = 332
    direct_pform = 4

    # Create the result folder if not exist
    if os.path.exists(saving_folder) :
        ()
    else :
        P = pathlib.Path(saving_folder)
        pathlib.Path.mkdir(P, parents = True)

    print('')
    print('#####       ')
    print('Start calibration')
    print('#####       ')

    all_Ucam, all_Xref, nb_pts = data.pattern_detection(__calibration_dict__)


    
    A111, A_pol, Magnification = Soloff_calibration (__calibration_dict__,
                                                     x3_list,
                                                     Soloff_pform)

    print('')
    print('#####       ')
    print('End calibration')
    print('#####       ')


    print('')
    print('#####       ')
    print('Direct method calibration')
    print('#####       ')
    print('')

    direct_A, Magnification = direct_calibration (__calibration_dict__,
                                                  x3_list,
                                                  direct_pform)

    Magnification_z = np.mean (Magnification, axis = 0)
    
    print('')
    print('#####       ')
    print('Identification by DIC')
    print('#####       ')
    print('')
    
    Xleft_idc, Xright_idc = data.DIC_3D_composed_detection(__DIC_dict__,
                                                           flip = False)
    
    Xleft_id, Xright_id = data.DIC_disflow(__DIC_dict__,
                                           flip = False)
    
    Np_img, Npoints, Naxes = Xleft_id.shape
    all_U, all_V, all_W = np.zeros((3, 2*Np_img, Npoints))
    xDirect_solutions = np.zeros((Np_img, 3, Npoints))
    xSoloff_solutions = np.zeros((Np_img, 3, Npoints))
    xAI_solutions = np.zeros((Np_img, 3, Npoints))

    for image in range (1) :
        print('')
        print('')
        print('Calculation of the DIC ', image)
        print('...')
        print('...')
        
        X_c1 = Xleft_id[image]
        X_c2 = Xright_id[image]
        
        # Create the DataFrame
        data_c = {'X1' :  X_c1[:,0], 'Y1' :  X_c1[:,1], 'X2' :  X_c2[:,0], 'Y2' :  X_c2[:,1]}
        df = pd.DataFrame(data = data_c)
        
        # Direct identification
        t0 = time.time()
        xDirect_solution = direct_identification (X_c1,
                                                  X_c2,
                                                  direct_A,
                                                  direct_pform)
        xD, yD, zD = xDirect_solution
        df.insert(df.shape[1], 'xDirect', xD, True)
        df.insert(df.shape[1], 'yDirect', yD, True)
        df.insert(df.shape[1], 'zDirect', zD, True)
        xDirect_solutions[image] = xDirect_solution
        t1 = time.time()
        print('time direct = ',t1 - t0)
        print('end direct')

        fit, er, mean_er, res = solvel.fit_plan_to_points(xDirect_solution)
        plt.figure()
        plt.show()

        # Soloff identification
        t0 = time.time()
        soloff_file = saving_folder + '/xsolution_soloff200_1800' + str(image) + '.npy'
        if os.path.exists(soloff_file) and True :
            xSoloff_solution = np.load(soloff_file)
        else :
            xSoloff_solution = Soloff_identification (X_c1,
                                                      X_c2,
                                                      A111, 
                                                      A_pol,
                                                      Soloff_pform,
                                                      method = 'curve_fit')       
            np.save(soloff_file, xSoloff_solution)
        sys.exit()
        t1 = time.time()
        print('time Soloff = ',t1 - t0)
        print('end SOLOFF')
        # Points coordinates
        xS, yS, zS = xSoloff_solution
        xSoloff_solutions[image] = xSoloff_solution
        fit, er, mean_er, res = solvel.fit_plan_to_points(xSoloff_solution)
        plt.figure()        
        plt.show()

        
        # Lagrangian projection on left image 
        win = __DIC_dict__['window']
        n_wx = win[0][1] - win[0][0]
        n_wy = win[1][1] - win[1][0]
        X = np.reshape(xS,(n_wx,n_wy))
        Y = np.reshape(yS,(n_wx,n_wy))
        Z = np.reshape(zS,(n_wx,n_wy))
        X0,Y0=np.meshgrid(np.linspace(X.min(),X.max(),1600),
                          np.linspace(Y.min(),Y.max(),1600))
        # plt.figure()
        # plt.imshow(np.gradient(Y-Y0)[1])
        # plt.title('Gradient Y')
        # plt.clim(-0.001,0.001)
        # cb = plt.colorbar()
        # cb.set_label('z in mm')
        # plt.colorbar()
        # plt.show()
        
        # plt.figure()
        # plt.imshow(np.gradient(X-X0)[1])
        # plt.title('Gradient X')
        # plt.clim(-0.001,0.001)
        # cb = plt.colorbar()
        # cb.set_label('z in mm')
        # plt.colorbar()
        # plt.show()        
        
        # plt.figure()
        # plt.imshow(Z)
        # plt.title('Z projected on left camera')
        # cb = plt.colorbar()
        # cb.set_label('z in mm')
        # plt.show()    

        # Chose the Number of Datas for Artificial Intelligence Learning
        AI_training_size = 50000
        # Create the .csv to make an AI identification
        AI_file = saving_folder + '/xsolution_AIxsolution_soloff200_1800' + str(image) + '.npy'
        
        x_, Xc1_, Xc2_ = data.camera_np_coordinates(all_Ucam, 
                                                    all_Xref, 
                                                    x3_list)
        
        Xc1_ = np.transpose(Xc1_)
        Xc2_ = np.transpose(Xc2_)
        
        t0 = time.time()
        if os.path.exists(AI_file) and False :
            xAI_solution = np.load(AI_file)
        else :
            # Train the AI with already known points
            if image == 0 :
                model_file = saving_folder +'/Soloff_AI_' + str(image) + '_' + str(AI_training_size) + '_points.csv'      
                model = AI_training (X_c1,
                                     X_c2,
                                     xSoloff_solution,
                                     AI_training_size = AI_training_size,
                                     file = model_file)
                
                model_file2 = saving_folder +'/Soloff_AI_' + str(image) + '_ChAruco.csv'      
                model2 = AI_training (Xc1_,
                                      Xc2_,
                                      x_,
                                      AI_training_size = AI_training_size,
                                      file = model_file2)
            t1 = time.time()
            print('time training = ',t1 - t0)

            # Use the AI model to solve every points
            xAI_solution = AI_identification (X_c1,
                                              X_c2,
                                              model)
            xAI_solution2 = AI_identification (X_c1,
                                               X_c2,
                                               model2)
            
            np.save(AI_file, xAI_solution)
            np.save(AI_file, xAI_solution2)
        t2 = time.time()
        print('time AI = ',t2 - t1)
        print('end AI')
        xAI, yAI, zAI = xAI_solution
        
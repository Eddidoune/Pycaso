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
             saving_folder,
             polynomial_form = 332,
             detection = True,
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
       saving_folder : str
           Folder to save datas
       polynomial_form : int, optional
           Polynomial form
       detection : bool, optional
           If True, all the analysis will be done. If False, the code will 
           take the informations in 'saving_folder'
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
           Constants of Soloff polynomial form chose (polynomial_form)
       Magnification : int
           Magnification between reals and detected positions 
           [[Mag Left x, Mag Left y], [Mag Right x, Mag Right y]]
    """
    
    A111 = np.zeros((2, 2, 4))
    if polynomial_form == 111 :
        A_pol = np.zeros((2, 2, 4))
    elif polynomial_form == 221 :
        A_pol = np.zeros((2, 2, 9))
    elif polynomial_form == 222 :
        A_pol = np.zeros((2, 2, 10))
    elif polynomial_form == 332 :
        A_pol = np.zeros((2, 2, 19))
    elif polynomial_form == 333 :
        A_pol = np.zeros((2, 2, 20))
    elif polynomial_form == 443 :
        A_pol = np.zeros((2, 2, 34))
    elif polynomial_form == 444 :
        A_pol = np.zeros((2, 2, 35))
    elif polynomial_form == 554 :
        A_pol = np.zeros((2, 2, 55))
    elif polynomial_form == 555 :
        A_pol = np.zeros((2, 2, 56))    
    else :
        print ('Only define for polynomial forms (111, 221, 222, 332, 333, 443, 444, 554 or 555')
        sys.exit()
    
    A_0 = [A111, A_pol]
    polynomial_forms = [111, polynomial_form]

    
    # Detect points from folders
    all_Ucam, all_Xref, nb_pts = data.pattern_detection(__calibration_dict__,
                                   detection = detection,
                                   saving_folder = saving_folder,
                                   hybrid_verification = hybrid_verification)        

    # Creation of the reference matrix Xref and the real position Ucam for 
    # each camera
    x, Xc1, Xc2 = data.camera_np_coordinates(all_Ucam, 
                                             all_Xref, 
                                             x3_list)

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
            polynomial_form = polynomial_forms[pol]
            M = solvel.Soloff_Polynome({'polynomial_form' : polynomial_form}).pol_form(x)
            Ai = np.matmul(X, np.linalg.pinv(M))
            A_0[pol][camera-1] = Ai
    
            # Error of projection
            Xd = np.matmul(Ai,M)
            proj_error = X - Xd
            print('Max ; min projection error (polynomial form ', 
                  str(polynomial_form),
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
                        polynomial_form = 332,
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
       polynomial_form : int, optional
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
                                                            polynomial_form = polynomial_form, 
                                                            method = 'curve_fit')
    return (x_solution)


def direct_calibration (__calibration_dict__,
             x3_list,
             saving_folder,
             direct_polynome_degree,
             detection = True) :
    """Calculation of the magnification between reals and detected positions 
    and the calibration parameters A:--> x = A.M(X)
    
    Args:
       __calibration_dict__ : dict
           Calibration properties define in a dict. Including 'left_folder', 
           'right_folder', 'name', 'ncx', 'ncy', 'sqr'
       x3_list : numpy.ndarray
           List of the different z position. (WARNING : Should be order the 
                                              same way in the target folder)
       saving_folder : str
           Folder to save datas
       direct_polynome_degree : int, optional
           Polynomial degree
       detection : bool, optional
           If True, all the analysis will be done. If False, the code will 
           take the informations in 'saving_folder'

    Returns:
       A : numpy.ndarray
           Constants of direct polynomial
       Magnification : int
           Magnification between reals and detected positions
    """
    
    if direct_polynome_degree == 1 :
        direct_A = np.zeros((3, 5))
    elif direct_polynome_degree == 2 :
        direct_A = np.zeros((3, 15))
    elif direct_polynome_degree == 3 :
        direct_A = np.zeros((3, 35))
    elif direct_polynome_degree == 4 :
        direct_A = np.zeros((3, 70))
    else :
        print ('Only define for polynomial degrees (1, 2, 3 or 4')
        sys.exit()
    # Detect points from folders
    all_Ucam, all_Xref, nb_pts = data.pattern_detection(__calibration_dict__, 
                                                        detection = detection, 
                                                        saving_folder = saving_folder)        

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
        M = solvel.Direct_Polynome({'polynomial_form' : direct_polynome_degree}).pol_form(Xc1, Xc2)
        Ap = np.matmul(x, np.linalg.pinv(M))
        direct_A = Ap

        # Error of projection
        xd = np.matmul(Ap,M)
        proj_error = x - xd
        print('Max ; min projection error (polynomial form ',
              str(direct_polynome_degree),
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
                           direct_polynomial_form = 3) :
    """Identification of the points detected on both cameras left and right 
    into the global 3D-space
    
    Args:
       Xc1_identified : numpy.ndarray
           Points identified on the left camera
       Xc2_identified : numpy.ndarray
           Points identified on the right camera
       direct_A : numpy.ndarray
           Constants of direct polynomial
       direct_polynomial_form : int, optional
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
    
    M = solvel.Direct_Polynome({'polynomial_form' : direct_polynomial_form}).pol_form(Xl, Xr)
    xsolution = np.matmul(direct_A,M)
    return(xsolution)

def IA_identification (X_c1,
                       X_c2,
                       xSoloff_solution,
                       NDIAL = 1000,
                       file = 'Soloff_IA_.csv') :
    """Calculation of the magnification between reals and detected positions 
    and the calibration parameters A:--> x = A.M(X)
    
    Args:
       X_c1 : numpy.ndarray
           Left coordinates of points.
       X_c2 : numpy.ndarray
           Right coordinates of points.
       xSoloff_solution : numpy.ndarray
           3D space coordinates identified with Soloff method.
       NDIAL : int
           Number of datas (points) used to train the IA metamodel.
       file : str
           Name of saving file for training

    Returns:
       xIA_solution : numpy.ndarray
           3D space coordinates identified with IA method.
    """
    # Organise the list of parameters
    Xl0, Yl0 = X_c1[:,0], X_c1[:,1]
    Xr0, Yr0 = X_c2[:,0], X_c2[:,1]
    xS, yS, zS = xSoloff_solution
    
    # List of random points (To minimize the size of calculation)
    rd_list = np.ndarray.astype((np.random.rand(NDIAL)*X_c1.shape[0]),int)    
    with open(file, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, 
                                delimiter=' ',
                                quotechar='|', 
                                quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['Xl'] + ['Yl'] + ['Xr'] + ['Yr'] + 
                            ['x'] + ['y'] + ['z'])
        for i in range (NDIAL) :
            spamwriter.writerow([str(Xl0[rd_list[i]]), 
                                 str(Yl0[rd_list[i]]), 
                                 str(Xr0[rd_list[i]]), 
                                 str(Yr0[rd_list[i]]), 
                                 str(xS[rd_list[i]]), 
                                 str(yS[rd_list[i]]), 
                                 str(zS[rd_list[i]])])
    # Build the IA model with the NDIAL.
    model, accuracy = solvel.AI_solve (file)
    # Solve on all pts
    X = np.transpose(np.array([X_c1[:,0], X_c1[:,1], 
                               X_c2[:,0], X_c2[:,1]]))        
    t0 = time.time()
    xIA_solution = model.predict(X)
    t1 = time.time()
    print('time IA = ',t1 - t0)
    print('end IA')
    xIA_solution = np.transpose(xIA_solution)
    return (xIA_solution)


if __name__ == '__main__' :
    main_path = "/home/caroneddy/These/Stereo_camera/Pycaso_archives/src"    

    # Define the inputs
    __calibration_dict__ = {
    'left_folder' : main_path + '/Images_example/2022_02_02/left-calib',
    'right_folder' : main_path + '/Images_example/2022_02_02/right-calib',
    'name' : 'micro_calibration',
    'ncx' : 16,
    'ncy' : 12,
    'sqr' : 0.3}  #in mm
    
    __pattern_dict__ = {
    'left_folder' : main_path + '/Images_example/2022_02_02/test_left',
    'right_folder' : main_path + '/Images_example/2022_02_02/test_right',
    'name' : 'micro_identification',
    'ncx' : 16,
    'ncy' : 12,
    'sqr' : 0.3,
    'window' : [[500, 1500], [500, 1500]]}  #in mm
    
    # __DIC_dict__ = {
    # 'left_folder' : main_path + '/Images_example/2022_04_15/left_sample_identification',
    # 'right_folder' : main_path + '/Images_example/2022_04_15/right_sample_identification',
    # 'name' : 'micro_identification',
    # 'window' : [[200, 1800], [200, 1800]]}  #in mm
    
    # Create the list of z plans
    Folder = __calibration_dict__['left_folder']
    Imgs = sorted(glob(str(Folder) + '/*'))
    x3_list = np.zeros((len(Imgs)))
    for i in range (len(Imgs)) :
        x3_list[i] = float(Imgs[i][len(Folder)+ 1:-4])
        
    saving_folder = main_path + '/results/2022_02_02_results'
    
    # Chose the polynomial degree for the calibration fitting
    polynomial_form = 332
    direct_polynomial_form = 4

    # Create the result folder if not exist
    if os.path.exists(saving_folder) :
        ()
    else :
        P = pathlib.Path(saving_folder)
        pathlib.Path.mkdir(P, parents = True)

    sys.exit()

    print('')
    print('#####       ')
    print('Start calibration')
    print('#####       ')

    all_Ucam, all_Xref, nb_pts = data.pattern_detection(__calibration_dict__,
                                                        detection = False,
                                                        saving_folder = saving_folder,
                                                        hybrid_verification = False)   
    
    A111, A_pol, Magnification = Soloff_calibration (__calibration_dict__,
                                                     x3_list,
                                                     saving_folder,
                                                     polynomial_form = polynomial_form,
                                                     detection = False)

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
                                                  saving_folder,
                                                  direct_polynomial_form,
                                                  detection = False)

    Magnification_z = np.mean (Magnification, axis = 0)
    
    print('')
    print('#####       ')
    print('Identification by DIC')
    print('#####       ')
    print('')
    
    X, x, nb_pts = data.pattern_detection(__pattern_dict__,
                                          detection = True,
                                          saving_folder = saving_folder,
                                          hybrid_verification = False)
    
    Xl, Xr = X[0], X[1]
    xDirect_solution = direct_identification (Xl,
                                              Xr,
                                              direct_A,
                                              direct_polynomial_form = direct_polynomial_form)
    xD, yD, zD = xDirect_solution

    
    
    xSoloff_solution = Soloff_identification (Xl,
                                              Xr,
                                              A111, 
                                              A_pol,
                                              polynomial_form = polynomial_form,
                                              method = 'curve_fit')   
    xS, yS, zS = xSoloff_solution

    fit, er, mean_er, res = solvel.fit_plan_to_points(xDirect_solution)
    zD_recal = zD - fit[0]*xD - fit[1]*yD - fit[2]
    plt.title("M??thode directe : Reconstruction 3D d'une image la mire")
    plt.figure()
    plt.show()    

    fit, er, mean_er, res = solvel.fit_plan_to_points(xSoloff_solution)
    zS_recal = zS - fit[0]*xS - fit[1]*yS - fit[2]   
    plt.title("M??thode de Soloff : Reconstruction 3D d'une image la mire")
    plt.figure()
    plt.show()    
    
    sys.exit()


    Xleft_id, Xright_id = data.DIC_3D_detection_lagrangian(__DIC_dict__, 
                                                           detection = False,
                                                           saving_folder = saving_folder,
                                                           flip = False)
    
    Np_img, Npoints, Naxes = Xleft_id.shape
    all_U, all_V, all_W = np.zeros((3, 2*Np_img, Npoints))
    xDirect_solutions = np.zeros((Np_img, 3, Npoints))
    xSoloff_solutions = np.zeros((Np_img, 3, Npoints))
    xIA_solutions = np.zeros((Np_img, 3, Npoints))
    for image in range (Np_img) :
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
                                                  direct_polynomial_form = direct_polynomial_form)
        xD, yD, zD = xDirect_solution
        df.insert(df.shape[1], 'xDirect', xD, True)
        df.insert(df.shape[1], 'yDirect', yD, True)
        df.insert(df.shape[1], 'zDirect', zD, True)
        xDirect_solutions[image] = xDirect_solution
        t1 = time.time()
        print('time direct = ',t1 - t0)
        print('end direct')
        
        # # Creating figure
        # fig = plt.figure(figsize = (16, 9))
        # ax = plt.axes(projection ="3d")
        # # Add x, y gridlines
        # ax.grid(visible = True, color ='grey',
        #         linestyle ='-.', linewidth = 0.3,
        #         alpha = 0.2)
        # # Creating color map
        # my_cmap = plt.get_cmap('hsv')
        # # Creating plot
        # sctt = ax.scatter3D(xD, yD, zD,
        #                     alpha = 0.8,
        #                     c = zD,
        #                     cmap = my_cmap)
        # plt.title("Direct all pts" + str(image))
        # ax.set_xlabel('x (mm)', fontweight ='bold')
        # ax.set_ylabel('y (mm)', fontweight ='bold')
        # ax.set_zlabel('z (mm)', fontweight ='bold')
        # fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5)
        # # show plot
        # plt.show()            
            

        # Soloff identification
        t0 = time.time()
        soloff_file = saving_folder + '/xsolution_soloff' + str(image) + '.npy'
        if os.path.exists(soloff_file) and True :
            xSoloff_solution = np.load(soloff_file)
        else :
            xSoloff_solution = Soloff_identification (X_c1,
                                                      X_c2,
                                                      A111, 
                                                      A_pol,
                                                      polynomial_form = polynomial_form,
                                                      method = 'curve_fit')       
            np.save(soloff_file, xSoloff_solution)

        t1 = time.time()
        print('time Soloff = ',t1 - t0)
        print('end SOLOFF')
        # Points coordinates
        xS, yS, zS = xSoloff_solution
        xSoloff_solutions[image] = xSoloff_solution
        fit, er, mean_er, res = solvel.fit_plan_to_points(xSoloff_solution)
        plt.figure()        
        plt.show()

        # zS_recal = zS - fit[0]*xS - fit[1]*yS - fit[2]       
        # df.insert(df.shape[1], 'xSoloff', xS, True)
        # df.insert(df.shape[1], 'ySoloff', yS, True)
        # df.insert(df.shape[1], 'zSoloff', zS_recal, True)        
        
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
        
        plt.figure()
        plt.imshow(Z)
        plt.title('Z projected on left camera')
        cb = plt.colorbar()
        cb.set_label('z in mm')
        plt.show()    

        '''
        
        
        plt.figure()
        plt.imshow(Z-nd.median_filter(Z,5))
        plt.title('Z median filter projected on left camera')
        cb = plt.colorbar()
        plt.clim(3.15, 3.27) 
        cb.set_label('z in mm')
        plt.clim(-0.0005,0.0005)
        plt.show()               
        
        # sys.exit()
        
        # Calcul of the cinematic fields
        if image != 0 :
            N_xy = int(math.sqrt(len(xS)))
            x0, y0, z0 = np.reshape(xSoloff_solutions[0],(3,N_xy,N_xy))
            xt, yt, zt = np.reshape(xSoloff_solution,(3,N_xy,N_xy))
            U, V, W = x0-xt, y0-yt, z0-zt
            df.insert(df.shape[1], 'U', np.ravel(U), True)
            df.insert(df.shape[1], 'V', np.ravel(V), True)
            df.insert(df.shape[1], 'W', np.ravel(W), True)
            # Plot the displacements fields (here U)
            Exy, Exx = np.gradient(U)
            Eyy, Eyx = np.gradient(V)
            Ezy, Ezx = np.gradient(W)
            Exx = 100 * Exx
            Eyy = 100 * Eyy
            Ezy = 100 * Ezy
            
            aa = 20
            plt.imshow(Exx,plt.get_cmap('hot'))
            cb = plt.colorbar()
            plt.clim(np.min(Exx)/aa,np.max(Exx)/aa)
            cb.set_label(r'Exx (%)')
            plt.title('Exx')
            plt.figure()
            plt.show()
            
            plt.imshow(Eyy,plt.get_cmap('hot'))
            cb = plt.colorbar()
            plt.clim(np.min(Eyy)/aa,np.max(Eyy)/aa)
            cb.set_label(r'Eyy (%)')
            plt.title('Eyy')
            plt.figure()
            plt.show()
            
            plt.imshow(Ezy,plt.get_cmap('hot'))
            cb = plt.colorbar()
            plt.clim(np.min(Ezy)/aa,np.max(Ezy)/aa)
            cb.set_label(r'Ezy (%)')
            plt.title('Ezy')
            plt.figure()
            plt.show()

            
        
        '''
        '''
        # Chose the Number of Datas for Artificial Intelligence Learning
        NDIAL = 1000
        # Create the .csv to make an IA identification
        file = saving_folder +'/15_Soloff_IA_' + str(image) + '.csv'      
        xIA_solution = IA_identification (X_c1,
                                          X_c2,
                                          xSoloff_solution,
                                          NDIAL = NDIAL,
                                          file = file)
        xIA, yIA, zIA = xIA_solution
        df.insert(df.shape[1], 'xIA', xIA, True)
        df.insert(df.shape[1], 'yIA', yIA, True)
        df.insert(df.shape[1], 'zIA', zIA, True)   
        xIA_solutions[image] = xIA_solution
        
        # Creating figure
        fig = plt.figure(figsize = (16, 9))
        ax = plt.axes(projection ="3d")
        ax.grid(visible = True, color ='grey',
                linestyle ='-.', linewidth = 0.3,
                alpha = 0.2)
        my_cmap = plt.get_cmap('hsv')
        sctt = ax.scatter3D(xIA, yIA, zIA,
                            alpha = 0.8,
                            c = zIA,
                            cmap = my_cmap)
        plt.title("IA all pts" + str(image))
        ax.set_xlabel('x (mm)', fontweight ='bold')
        ax.set_ylabel('y (mm)', fontweight ='bold')
        ax.set_zlabel('z (mm)', fontweight ='bold')
        fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5)
        plt.show()
        '''
        
        
        
        
        
        # # Solve on 1000 pts to compare with Soloff
        # X_compare = np.transpose(np.array([Xl[:,0], Xl[:,1], Xr[:,0], Xr[:,1]]))
        # IA_compare = model.predict(X_compare)
        # xIA_compare, yIA_compare, zIA_compare = np.transpose(IA_compare)
        
        
        
        # # Difference IA and Soloff
        # xdiff, ydiff, zdiff = xs - xIA_compare, ys - yIA_compare, zs - zIA_compare
        # r = np.sqrt(xdiff**2 + ydiff**2 + zdiff**2)
        # # Creating figure
        # fig = plt.figure(figsize = (16, 9))
        # ax = plt.axes(projection ="3d")
        # sctt = ax.scatter3D(xdiff, ydiff, zdiff,
        #                     alpha = 0.8,
        #                     c = r)
        # plt.title("Soloff/IA diff" + str(image))
        # ax.set_xlabel('x (mm)', fontweight ='bold')
        # ax.set_ylabel('y (mm)', fontweight ='bold')
        # ax.set_zlabel('z (mm)', fontweight ='bold')
        # fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5)
        # plt.show()
        # print('max Soloff/IA diff x, y, z, r : ', np.around(np.max(xdiff), 4), np.around(np.max(ydiff), 4), np.around(np.max(zdiff), 4), np.around(np.max(r), 4))
        # print('mean Soloff/IA diff x, y, z, r : ', np.around(np.mean(np.abs(xdiff)), 4), np.around(np.mean(np.abs(ydiff)), 4), np.around(np.mean(np.abs(zdiff)), 4), np.around(np.mean(np.abs(r)), 4))
        # print('std Soloff/IA diff x, y, z, r : ', np.around(np.std(xdiff), 4), np.around(np.std(ydiff), 4), np.around(np.std(zdiff), 4), np.around(np.std(r), 4))
         
        

        # # Difference IA and direct
        # xdiff, ydiff, zdiff = xD - xIA, yD - yIA, zD - zIA
        # r = np.sqrt(xdiff**2 + ydiff**2 + zdiff**2)
        # # Creating figure
        # fig = plt.figure(figsize = (16, 9))
        # ax = plt.axes(projection ="3d")
        # sctt = ax.scatter3D(xdiff, ydiff, zdiff,
        #                     alpha = 0.8,
        #                     c = r)
        # plt.title("direct/IA diff" + str(image))
        # ax.set_xlabel('x (mm)', fontweight ='bold')
        # ax.set_ylabel('y (mm)', fontweight ='bold')
        # ax.set_zlabel('z (mm)', fontweight ='bold')
        # fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5)
        # plt.show()
        # print('max direct/IA diff x, y, z, r : ', np.around(np.max(xdiff), 4), np.around(np.max(ydiff), 4), np.around(np.max(zdiff), 4), np.around(np.max(r), 4))
        # print('mean direct/IA diff x, y, z, r : ', np.around(np.mean(np.abs(xdiff)), 4), np.around(np.mean(np.abs(ydiff)), 4), np.around(np.mean(np.abs(zdiff)), 4), np.around(np.mean(np.abs(r)), 4))
        # print('std direct/IA diff x, y, z, r : ', np.around(np.std(xdiff), 4), np.around(np.std(ydiff), 4), np.around(np.std(zdiff), 4), np.around(np.std(r), 4))

        # print('')
        # print('#####       ')
        # print('End coin test identification')
        # print('#####       ')
        # print('')
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 09:19:07 2021

@author: Eddidoune
"""
import sigfig as sgf
try : 
    import cupy 
    cpy = True
except ImportError:
    cpy = False
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import solve_library as solvel 
import data_library as data
import csv
import math
import matplotlib.pyplot as plt
import os
import pathlib

def magnification (X1 : np.ndarray, 
                   X2 : np.ndarray,
                   x1 : np.ndarray,
                   x2 : np.ndarray) -> np.ndarray :
    """Calculation of the magnification between reals and detected positions
    in unity(mm or µm)/px
    
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
       Magnification : numpy.ndarray
           Magnification between detected and real positions
           [Mag x, Mag y]
    """
    Delta_X1 = np.nanmean(abs(X1-np.nanmean(X1)))
    Delta_X2 = np.nanmean(abs(X2-np.nanmean(X2)))
    Delta_x1 = np.nanmean(abs(x1-np.nanmean(x1)))
    Delta_x2 = np.nanmean(abs(x2-np.nanmean(x2)))
    Magnification = np.asarray([Delta_x1/Delta_X1, Delta_x2/Delta_X2]) 
    return (Magnification)

def Soloff_identification (Xc1_identified : np.ndarray,
                           Xc2_identified : np.ndarray,
                           Soloff_constants0 : np.ndarray, 
                           Soloff_constants : np.ndarray,
                           Soloff_pform : int,
                           method : str = 'Peter',
                           cut : int = 0) -> np.ndarray :
    """Identification of the points detected on both cameras cam1 and cam2 
    into the global 3D-space
    
    Args:
       Xc1_identified : numpy.ndarray
           Points identified on the cam1 camera
       Xc2_identified : numpy.ndarray
           Points identified on the cam2 camera
       Soloff_constants0 : numpy.ndarray
           Constants of Soloff polynomial form '111'
       Soloff_constants : numpy.ndarray
           Constants of Soloff polynomial form chose (polynomial_form)
       Soloff_pform : int
           Polynomial form
       method : str, optional
           Python method used to solve it ('Peter' 'curve-fit')
       cut : int, optional
           Used by Peter method to reduce the windows of resolution

    Returns:
       x_solution : numpy.ndarray
           Identification in the 3D space of the detected points
    """
    if method == 'Peter' :
        if Soloff_pform != 332 :
            raise('No Peter resolution for the polynomial form ', Soloff_pform, ' for the moment (Only for the form 332')
        Xc1 = np.copy(Xc1_identified[:,:,0])
        Yc1 = np.copy(Xc1_identified[:,:,1])
        Xc2 = np.copy(Xc2_identified[:,:,0])
        Yc2 = np.copy(Xc2_identified[:,:,1])
        Xc10 = []
        Yc10 = []
        Xc20 = []
        Yc20 = []

        l = 20
        nx,ny = Xc1.shape
        nx = nx//l
        ny = ny//l
        for i in range (l) :
            for j in range (l) :
                Xc10.append(Xc1[i*nx, j*ny])
                Yc10.append(Yc1[i*nx, j*ny])
                Xc20.append(Xc2[i*nx, j*ny])
                Yc20.append(Yc2[i*nx, j*ny])
        
        Xc10 = np.array(Xc10).reshape((l,l))
        Yc10 = np.array(Yc10).reshape((l,l))
        Xc20 = np.array(Xc20).reshape((l,l))
        Yc20 = np.array(Yc20).reshape((l,l))
        Xc1_identified0 = np.zeros((l,l,2))
        Xc1_identified0[:,:,0] = Xc10
        Xc1_identified0[:,:,1] = Yc10
        Xc2_identified0 = np.zeros((l,l,2))
        Xc2_identified0[:,:,0] = Xc20
        Xc2_identified0[:,:,1] = Yc20        
        x,y,z = Soloff_identification (Xc1_identified0,
                                       Xc2_identified0,
                                       Soloff_constants0, 
                                       Soloff_constants,
                                       Soloff_pform,
                                       method = 'curve_fit') 
        print('')
        print('')
        print('Peter mapping')
        print('xmax',np.max(x))
        print('ymax',np.max(y))
        print('zmax',np.max(z))
        print('xmin',np.min(x))
        print('ymin',np.min(y))
        print('zmin',np.min(z))
        print('')
        print('')
        xsolution = solvel.Peter(Xc1,
                                 Yc1,
                                 Xc2,
                                 Yc2,
                                 Soloff_constants,
                                 x,
                                 y,
                                 z,
                                 cut = cut)
    
    else :
        if len(Xc1_identified.shape) == 3 : 
            modif_22_12_09 = True
            nx, ny, naxis = Xc1_identified.shape
            Xc1_identified = Xc1_identified.reshape((nx*ny, naxis))
            Xc2_identified = Xc2_identified.reshape((nx*ny, naxis))
        elif len(Xc1_identified.shape) == 2 : 
            modif_22_12_09 = False
        else :
            raise('Error, X_ci shape different than 2 or 3')
        # We're searching for the solution x0(x1, x2, x3) as Xc1 = ac1 . 
        # (1 x1 x2 x3) and Xc2 = ac2 . (1 x1 x2 x3)  using least square method.
        x0 = solvel.least_square_method (Xc1_identified, 
                                         Xc2_identified, 
                                         Soloff_constants0)
        
        # Solve the polynomials constants ai with curve-fit method (Levenberg 
        # Marcquardt)
        xsolution, Xc, Xd = solvel.Levenberg_Marquardt_solving(Xc1_identified, 
                                                               Xc2_identified, 
                                                               Soloff_constants, 
                                                               x0, 
                                                               Soloff_pform, 
                                                               method = method)
        if modif_22_12_09 :
            xsolution = xsolution.reshape((3, nx, ny))
    return (xsolution)

def direct_identification (Xc1_identified : np.ndarray,
                           Xc2_identified : np.ndarray,
                           direct_constants : np.ndarray,
                           direct_pform : int) -> np.ndarray :
    """Identification of the points detected on both cameras cam1 and cam2 
    into the global 3D-space
    
    Args:
       Xc1_identified : numpy.ndarray
           Points identified on the cam1 camera
       Xc2_identified : numpy.ndarray
           Points identified on the cam2 camera
       direct_constants : numpy.ndarray
           Constants of direct polynomial
       direct_pform : int
           Polynomial form

    Returns:
       x_solution : numpy.ndarray
           Identification in the 3D space of the detected points
    """    
    # Solve by direct method
    if len(Xc1_identified.shape) == 3 : 
        modif_22_12_09 = True
        nx, ny, naxis = Xc1_identified.shape
        Xc1_identified = Xc1_identified.reshape((nx*ny, naxis))
        Xc2_identified = Xc2_identified.reshape((nx*ny, naxis))
    elif len(Xc1_identified.shape) == 2 : 
        modif_22_12_09 = False
    else :
        raise('Error, X_ci shape different than 2 or 3')
    Xc11, Xc12 = Xc1_identified[:,0], Xc1_identified[:,1]
    Xc21, Xc22 = Xc2_identified[:,0], Xc2_identified[:,1]
    Xc1 = np.zeros((2,len(Xc11)))
    Xc2 = np.zeros((2,len(Xc21)))
    Xc1 = Xc11, Xc12
    Xc2 = Xc21, Xc22
    
    M = solvel.Direct_Polynome({'polynomial_form' : direct_pform}).pol_form(Xc1, Xc2)
    xsolution = np.matmul(direct_constants,M)
    if modif_22_12_09 :
        xsolution = xsolution.reshape((3, nx, ny))
    return(xsolution)

def Zernike_identification (Xc1_identified : np.ndarray,
                            Xc2_identified : np.ndarray,
                            Zernike_constants : np.ndarray,
                            Zernike_pform : int,
                            Cameras_dimensions : list) -> np.ndarray :
    """Identification of the points detected on both cameras cam1 and cam2 
    into the global 3D-space
    
    Args:
       Xc1_identified : numpy.ndarray
           Points identified on the cam1 camera
       Xc2_identified : numpy.ndarray
           Points identified on the cam2 camera
       Zernike_constants : numpy.ndarray
           Constants of Zernike polynomial
       Zernike_pform : int
           Polynomial form

    Returns:
       x_solution : numpy.ndarray
           Identification in the 3D space of the detected points
    """    
    # Solve by Zernike method
    if len(Xc1_identified.shape) == 3 : 
        modif_22_12_09 = True
        nx, ny, naxis = Xc1_identified.shape
        Xc1_identified = Xc1_identified.reshape((nx*ny, naxis))
        Xc2_identified = Xc2_identified.reshape((nx*ny, naxis))
    elif len(Xc1_identified.shape) == 2 : 
        modif_22_12_09 = False
    else :
        raise('Error, X_ci shape different than 2 or 3')
    Xc11, Xc12 = Xc1_identified[:,0], Xc1_identified[:,1]
    Xc21, Xc22 = Xc2_identified[:,0], Xc2_identified[:,1]
    Xc1 = np.zeros((2,len(Xc11)))
    Xc2 = np.zeros((2,len(Xc21)))
    Xc1 = Xc11, Xc12
    Xc2 = Xc21, Xc22
    
    M = solvel.Zernike_Polynome({'polynomial_form' : Zernike_pform}).pol_form(Xc1, Xc2, Cameras_dimensions)
    xsolution = np.matmul(Zernike_constants,M)
    if modif_22_12_09 :
        xsolution = xsolution.reshape((3, nx, ny))
    return(xsolution)

def Soloff_Zernike_identification (Xc1_identified : np.ndarray,
                                   Xc2_identified : np.ndarray,
                                   Soloff_constants0 : np.ndarray, 
                                   Soloff_constants : np.ndarray,
                                   Soloff_pform : int,
                                   Cameras_dimensions : list,
                                   Zernike_pform : int = 3,
                                   method : str = 'Peter',
                                   cut : int = 0) -> np.ndarray :
    """Identification of the points detected on both cameras cam1 and cam2 
    into the global 3D-space
    
    Args:
       Xc1_identified : numpy.ndarray
           Points identified on the cam1 camera
       Xc2_identified : numpy.ndarray
           Points identified on the cam2 camera
       Soloff_constants0 : numpy.ndarray
           Constants of Soloff polynomial form '111'
       Soloff_constants : numpy.ndarray
           Constants of Soloff polynomial form chose (polynomial_form)
       Soloff_pform : int
           Polynomial form
       method : str, optional
           Python method used to solve it ('Peter' 'curve-fit')
       cut : int, optional
           Used by Peter method to reduce the windows of resolution

    Returns:
       x_solution : numpy.ndarray
           Identification in the 3D space of the detected points
    """
    if method == 'Peter' :
        if Soloff_pform != 332 :
            raise('No Peter resolution for the polynomial form ', Soloff_pform, ' for the moment (Only for the form 332')
        Xc1 = np.copy(Xc1_identified[:,:,0])
        Yc1 = np.copy(Xc1_identified[:,:,1])
        Xc2 = np.copy(Xc2_identified[:,:,0])
        Yc2 = np.copy(Xc2_identified[:,:,1])
        Xc10 = []
        Yc10 = []
        Xc20 = []
        Yc20 = []

        l = 20
        nx,ny = Xc1.shape
        nx = nx//l
        ny = ny//l
        for i in range (l) :
            for j in range (l) :
                Xc10.append(Xc1[i*nx, j*ny])
                Yc10.append(Yc1[i*nx, j*ny])
                Xc20.append(Xc2[i*nx, j*ny])
                Yc20.append(Yc2[i*nx, j*ny])
        
        Xc10 = np.array(Xc10).reshape((l,l))
        Yc10 = np.array(Yc10).reshape((l,l))
        Xc20 = np.array(Xc20).reshape((l,l))
        Yc20 = np.array(Yc20).reshape((l,l))
        Xc1_identified0 = np.zeros((l,l,2))
        Xc1_identified0[:,:,0] = Xc10
        Xc1_identified0[:,:,1] = Yc10
        Xc2_identified0 = np.zeros((l,l,2))
        Xc2_identified0[:,:,0] = Xc20
        Xc2_identified0[:,:,1] = Yc20        
        x,y,z = Soloff_identification (Xc1_identified0,
                                       Xc2_identified0,
                                       Soloff_constants0, 
                                       Soloff_constants,
                                       Soloff_pform,
                                       method = 'curve_fit') 
        print('')
        print('')
        print('Peter mapping')
        print('xmax',np.max(x))
        print('ymax',np.max(y))
        print('zmax',np.max(z))
        print('xmin',np.min(x))
        print('ymin',np.min(y))
        print('zmin',np.min(z))
        print('')
        print('')
        xsolution = solvel.Peter(Xc1,
                                 Yc1,
                                 Xc2,
                                 Yc2,
                                 Soloff_constants,
                                 x,
                                 y,
                                 z,
                                 cut = cut)
    
    else :
        if len(Xc1_identified.shape) == 3 : 
            modif_22_12_09 = True
            nx, ny, naxis = Xc1_identified.shape
            Xc1_identified = Xc1_identified.reshape((nx*ny, naxis))
            Xc2_identified = Xc2_identified.reshape((nx*ny, naxis))
        elif len(Xc1_identified.shape) == 2 : 
            modif_22_12_09 = False
        else :
            raise('Error, X_ci shape different than 2 or 3')
        # We're searching for the solution x0(x1, x2, x3) as Xc1 = ac1 . 
        # (1 x1 x2 x3) and Xc2 = ac2 . (1 x1 x2 x3)  using least square method.
        x0 = solvel.least_square_method (Xc1_identified, 
                                         Xc2_identified, 
                                         Soloff_constants0)
        
        # Solve the polynomials constants ai with curve-fit method (Levenberg 
        # Marcquardt)
        xsolution, Xc, Xd = solvel.Levenberg_Zernike_solving(Xc1_identified, 
                                                             Xc2_identified, 
                                                             Soloff_constants, 
                                                             x0, 
                                                             Soloff_pform, 
                                                             method = method)
        if modif_22_12_09 :
            xsolution = xsolution.reshape((3, nx, ny))
    return (xsolution)

def retroprojection_error (calibration_method : str,
                           pform : int,
                           calibration_constants : np.ndarray,
                           z_points : np.ndarray,
                           all_X : np.ndarray,
                           all_xth : np.ndarray,
                           cam1_folder : str = 'cam1_calibration',
                           cam2_folder : str = 'cam2_calibration',
                           name : str = 'calibration',
                           saving_folder : str = 'results',
                           ncx : int = 16,
                           ncy : int = 12,
                           sqr : float = 0.3,
                           hybrid_verification : bool = False,
                           save : bool = True,
                           eject_crown : int = 0,
                           method = 'curve-fit') -> (np.ndarray,
                                                     np.ndarray) :
    """Calcul of the retroprojection error between the theoratical plans 
    positions and the calculated plans (with pycaso_identification)
    
    Args:
       calibration_method : str
           Calibration method of your choice (Soloff, direct or Zernike)
       pform : int
           Polynomial form of the method.
       calibration_constants : np.ndarray (or tuple for Soloff calibration_method)
           Calibration's constants of the method. Tuple for Soloff method (A111,A_pol)
       z_points : numpy.ndarray
           List of the different z position
       all_X : numpy.ndarray
           The corners of the pattern detect by the camera ranged in an array 
           arrange with all cam1 pictures followed by all cam2 pictures. 
           Expl : [cam1_picture_1, cam1_picture_2, cam2_picture_1, 
                   cam2_picture_2]
       all_xth : numpy.ndarray
           The theoretical corners of the pattern
       ncx : int, optional
           The number of squares for the chessboard through x direction
       ncy : int, optional
           The number of squares for the chessboard through y direction
       method : str, optional
           Python method used to solve Soloff ('Peter' 'curve-fit')
       Other arguments are useless...
           
    Returns:
       z_mean : list
           mean position of each z retro calculated
       z_std : list
           standard deviation of each z retro calculated
       z_iter : list
           z new position after iteration (fit with a plan)
    """
    z_mean = []
    z_std = []
    z_iter = []
    n,npts,_ = all_X.shape
    n = n//2
    Cameras_dimensions = data.cameras_size(cam1_folder,
                                           cam2_folder)
    for i in range (n) :
        Xc1 = all_X[i]
        Xc2 = all_X[i+n]
        nx, ny = ncx-1-2*eject_crown, ncy-1-2*eject_crown
        Xc1 = np.reshape(Xc1,(ny, nx, 2))
        Xc2 = np.reshape(Xc2,(ny, nx, 2))
        ny,nx,_ = Xc1.shape
        if calibration_method == 'Zernike' :
            Zernike_A = calibration_constants
            Zernike_pform = pform
            Solution = Zernike_identification (Xc1,
                                               Xc2,
                                               Zernike_A,
                                               Zernike_pform,
                                               Cameras_dimensions)
        elif calibration_method == 'Soloff' :
            A111, A_pol = calibration_constants
            Soloff_pform = pform
            Solution = Soloff_identification (Xc1,
                                                  Xc2,
                                                  A111, 
                                                  A_pol,
                                                  Soloff_pform,
                                                  method = method)
        elif calibration_method == 'direct' :
            direct_A = calibration_constants
            direct_pform = pform
            Solution = direct_identification (Xc1,
                                                  Xc2,
                                                  direct_A,
                                                  direct_pform)
        

        z_theoric = np.reshape(z_points[i],(ny,nx))
        fit, errors, mean_error, residual = solvel.fit_plan_to_points(Solution, 
                                                                      title = 'Calibration plans',
                                                                      plotting = False)
        z_fit = z_points[i] - fit[0] * all_xth[i,:,0] - fit[1] * all_xth[i,:,1] - fit[2]
        z_fit = np.ravel(z_theoric) - z_fit
        
        # plt.figure()
        # plt.imshow(np.reshape(np.ravel(Solution[2]), (ny,nx)), plt.get_cmap('hot'), alpha=1)
        # cb = plt.colorbar()
        # cb.set_label('z (mm)')
        # plt.title('z distribution on plan = ' +str(i)+ ' ; iteration = ' +str(it) + ' ; method = ' + calibration_method)
        # plt.savefig(saving_folder +'/Pycaso_retroprojection_error/iteration' +str(it)+ '_z_plan' +str(i)+'_fit.pdf', dpi = 500)
        # plt.close()
        
        z_mean.append(np.mean(Solution[2]))
        z_std.append(np.std(Solution[2]-z_theoric))
        z_iter.append(z_fit)
    
    z_mean = np.asarray(z_mean)
    z_std = np.asarray(z_std)
    z_iter = np.asarray(z_iter)
    
    return(z_mean, z_std, z_iter)

def Soloff_calibration (z_list : np.ndarray,
                        Soloff_pform : int,
                        multifolder : bool = False,
                        plotting : bool = False,
                        iterations = 1,
                        **kwargs) -> (np.ndarray, 
                                      np.ndarray, 
                                      np.ndarray):
    """Calculation of the magnification between reals and detected positions 
    and the calibration parameters A = Soloff_constants0 (Resp Soloff_constants):--> X = A.M(x)
    
    Args:
       z_list : numpy.ndarray
           List of the different z position. (WARNING : Should be order the 
                                              same way in the target folder)
       Soloff_pform : int
           Polynomial form
       multifolder : bool, optional
           Used for specific image acquisition when all directions moved
       plotting = Bool
           Plot the calibration view or not
       iterations : int
           Number of iterations. For each of them, the z_list is changed 
           with the retroprojection error assuming that the set-up is not 
           perfect
        **kwargs : All the arguments of the fonction data.pattern_detection
           
    Returns:
       Soloff_constants0 : numpy.ndarray
           Constants of Soloff polynomial form '111'
       Soloff_constants : numpy.ndarray
           Constants of Soloff polynomial form chose (Soloff_pform)
       Magnification : numpy.ndarray
           Magnification between reals and detected positions 
           [[Mag cam1 x, Mag cam1 y], [Mag cam2 x, Mag cam2 y]]
    """

    if not Soloff_pform in [111, 1, 221, 222, 2, 332, 333, 3, 443, 444, 4, 554, 555, 5] :
        raise('Only define for polynomial forms 111, 221, 222, 332, 333, 443, 444, 554 or 555')
    
    Soloff_pforms = [1, Soloff_pform]
    
    z_list = np.array(z_list)
    try :
        save_retro = kwargs['saving_folder']+'/Pycaso_retroprojection_error/'
    except :
        save_retro = 'Pycaso_retroprojection_error'
    if not os.path.exists(save_retro) :
        P = pathlib.Path(save_retro)
        pathlib.Path.mkdir(P, parents = True)
    z_recover = z_list*1
    try :    
        for it in range(iterations) :
            
            if it == 0 :
                # Detect points from folders
                if multifolder :
                    all_X, all_xth, nb_pts = data.multifolder_pattern_detection(**kwargs)          
                else :
                    all_X, all_xth, nb_pts = data.pattern_detection(**kwargs)        
        
                # Using not well detected images, remove corresponding arrays from all_X 
                # (cam2 and cam1) and z_list
                nz,npts,_ = all_X.shape
                a = np.where(np.isnan(all_X[:,0,0]) == True)[0]
                b, c = [], []
                for i in a :
                    b.append(i)
                    if i < nz//2 :
                        b.append(i+nz//2)
                        c.append(i)
                    else :
                        b.append(i-nz//2)
                        c.append(i-nz//2)
                
                b = list(set(b))
                c = list(set(c))
                b.sort()
                c.sort()
                z_list = np.delete(z_list, c, axis = 0)
                all_xth = np.delete(all_xth, b, axis = 0)
                all_X = np.delete(all_X, b, axis = 0)
            
                # Creation of the reference matrix Xref and the real position Ucam for 
                # each camera
                nz, npts, _ = all_xth.shape
                z_points = np.ones((nz//2, npts))
                for i in range(nz//2) :
                    z_points[i] = z_points[i]*z_list[i]
            
            Soloff_constants0 = []
            Soloff_constants = []
            
            # Creation of the reference matrix Xref and the real position Ucam for 
            # each camera
            x, Xc1, Xc2 = data.camera_coordinates(all_X, all_xth, z_points)     
        
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
                
                for pol in range (2) :
                    # Do the system X = Ai*M, where M is the monomial of the real 
                    # coordinates of crosses and X the image coordinates, and M the 
                    # unknow (polynomial form aab)
                    Soloff_pform = Soloff_pforms[pol]
                    M = solvel.Soloff_Polynome({'polynomial_form' : Soloff_pform}).pol_form(x)
                    Ai = np.matmul(X, np.linalg.pinv(M))
                    if pol == 0 :
                        Soloff_constants0.append(Ai)
                    else :
                        Soloff_constants.append(Ai)
            px = 1/np.mean(Magnification)
                
            # Find and eventually plot the references plans
            if plotting :
                fit, errors, mean_error, residual = solvel.refplans(x, z_list, plotting = True)
                
            Soloff_constants0 = np.asarray(Soloff_constants0)
            Soloff_constants = np.asarray(Soloff_constants)
            
            #Calcul the retroprojection error
            try :
                z_mean, z_std, z_iter = retroprojection_error('Soloff',
                                                               Soloff_pform,
                                                               (Soloff_constants0,Soloff_constants),
                                                               z_points,
                                                               all_X,
                                                               all_xth,
                                                               **kwargs)
                mz_points = np.mean(z_points, axis=1)
                z_error = (z_mean-mz_points)*px
                
                plt.errorbar(mz_points, z_error, (z_std)*px, linestyle='None', marker='^')
                # plt.title('Soloff retroprojection error for iteration '+str(it+1))
                # plt.xlabel('z theoretical (mm)')
                plt.title("Erreur de rétroprojection (Soloff) : iteration N°"+str(it+1), size=17)
                plt.xlabel('z théorique (mm)', size=17)
                plt.ylabel('$\Delta$z (px)', size=17)
                plt.savefig(save_retro+'Retroprojection_error_Soloff_iteration'+str(it+1)+'.pdf', dpi = 500)
                plt.close()
                            
                # Save the z list
                np.save(save_retro+"z_list_Soloff.npy", z_points)
                
                # Change the list to the new ajusted
                z_points = z_iter
            except :
                print('Retroprojection estimation not possible : The ChAruco pattern might be partially out of field of view')
                z_error = np.array([0,0])
                
        # Error of projection
        print ('DEPTH OF FIELD :\n\t The calibrated depth of field is between \n\t', 
               np.min(z_list), 'mm and', np.max(z_list), 'mm.\n')    
        
        print('RETROPROJECTION ERROR : \n\t Max ; min (polynomial form', 
              str(Soloff_pform),') \n\t =',
              str(sgf.round(np.nanmax(z_error), sigfigs =3)),';',
              str(sgf.round(np.nanmin(z_error), sigfigs =3)),'px\n\t =',
              str(sgf.round(np.nanmax(z_error/px), sigfigs =3)),';',
              str(sgf.round(np.nanmin(z_error/px), sigfigs =3)),'mm')
    except :
        z_list = z_recover
        print('No iterations possible')
        # Detect points from folders
        if multifolder :
            all_X, all_xth, nb_pts = data.multifolder_pattern_detection(**kwargs)          
        else :
            all_X, all_xth, nb_pts = data.pattern_detection(**kwargs)
    
        # Creation of the reference matrix Xref and the real position Ucam for 
        # each camera
        nz, npts, _ = all_xth.shape
        z_points = np.ones((nz//2, npts))
        for i in range(nz//2) :
            z_points[i] = z_points[i]*z_list[i]

        Soloff_constants0 = []
        Soloff_constants = []
        
        # Creation of the reference matrix Xref and the real position Ucam for 
        # each camera
        x, Xc1, Xc2 = data.camera_coordinates(all_X, all_xth, z_points)     
    
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
            
            for pol in range (2) :
                # Do the system X = Ai*M, where M is the monomial of the real 
                # coordinates of crosses and X the image coordinates, and M the 
                # unknow (polynomial form aab)
                Soloff_pform = Soloff_pforms[pol]
                M = solvel.Soloff_Polynome({'polynomial_form' : Soloff_pform}).pol_form(x)
                Ai = np.matmul(X, np.linalg.pinv(M))
                if pol == 0 :
                    Soloff_constants0.append(Ai)
                else :
                    Soloff_constants.append(Ai)
        px = 1/np.mean(Magnification)
            
        # Find and eventually plot the references plans
        if plotting :
            fit, errors, mean_error, residual = solvel.refplans(x, z_list, plotting = True)
            
        Soloff_constants0 = np.asarray(Soloff_constants0)
        Soloff_constants = np.asarray(Soloff_constants)
    return(Soloff_constants0, Soloff_constants, Magnification)

def direct_calibration (z_list : np.ndarray,
                        direct_pform : int,
                        multifolder : bool = False,
                        plotting : bool = False,
                        iterations = 1,
                        **kwargs) -> (np.ndarray, 
                                      np.ndarray) :
    """Calculation of the magnification between reals and detected positions 
    and the calibration parameters A:--> x = A.M(X)
    
    Args:
       z_list : numpy.ndarray
           List of the different z position. (WARNING : Should be order the 
                                              same way in the target folder)
       direct_pform : int
           Polynomial degree
       multifolder : bool, optional
           Used for specific image acquisition when all directions moved
       plotting = Bool
           Plot the calibration view or not
       iterations : int
           Number of iterations. For each of them, the z_list is changed 
           with the retroprojection error assuming that the set-up is not 
           perfect
       **kwargs : All the arguments of the fonction data.pattern_detection
       
           
    Returns:
       direct_constants : numpy.ndarray
           Constants of direct polynomial
       Magnification : numpy.ndarray
           Magnification between reals and detected positions
    """
    # Test the constant form
    if not direct_pform in range(1, 5) :
        raise ('Only define for polynomial degrees (1, 2, 3, 4 or 5')
        
    z_list = np.array(z_list)
    try :
        save_retro = kwargs['saving_folder']+'/Pycaso_retroprojection_error/'
    except :
        save_retro = 'Pycaso_retroprojection_error/'
    if not os.path.exists(save_retro) :
        P = pathlib.Path(save_retro)
        pathlib.Path.mkdir(P, parents = True)
    z_recover = z_list*1
    try :
        for it in range(iterations) :
            if it == 0 :
                # Detect points from folders
                if multifolder :
                    all_X, all_xth, nb_pts = data.multifolder_pattern_detection(**kwargs)          
                else :
                    all_X, all_xth, nb_pts = data.pattern_detection(**kwargs)        
        
                # Using not well detected images, remove corresponding arrays from all_X 
                # (cam2 and cam1) and z_list
                nz,npts,_ = all_X.shape
                a = np.where(np.isnan(all_X[:,0,0]) == True)[0]
                b, c = [], []
                for i in a :
                    b.append(i)
                    if i < nz//2 :
                        b.append(i+nz//2)
                        c.append(i)
                    else :
                        b.append(i-nz//2)
                        c.append(i-nz//2)
                
                b = list(set(b))
                c = list(set(c))
                b.sort()
                c.sort()
                z_list = np.delete(z_list, c, axis = 0)
                all_xth = np.delete(all_xth, b, axis = 0)
                all_X = np.delete(all_X, b, axis = 0)

            
                # Creation of the reference matrix Xref and the real position Ucam for 
                # each camera
                nz, npts, _ = all_xth.shape
                z_points = np.ones((nz//2, npts))
                for i in range(nz//2) :
                    z_points[i] = z_points[i]*z_list[i]
            
            # Creation of the reference matrix Xref and the real position Ucam for 
            # each camera i
            x, Xc1, Xc2 = data.camera_coordinates(all_X, all_xth, z_points)
        
            # Calcul of the direct polynome's constants. X = A . M
            # Do the system x = Ap*M, where M is the monomial of the real 
            # coordinates of crosses and x the image coordinates, and M the unknow
            M = solvel.Direct_Polynome({'polynomial_form' : direct_pform}).pol_form(Xc1, Xc2)
            Ap = np.matmul(x, np.linalg.pinv(M))
            direct_constants = np.asarray(Ap)
            
            # Find and eventually plot the references plans
            if plotting :
                fit, errors, mean_error, residual = solvel.refplans(x, z_list, plotting = True)
            
            # Compute the magnification (same for each cam as set up is symetric)
            Magnification = np.zeros((2, 2))
            for camera in [1, 2] :
                if camera == 1 :
                    X = Xc1
                elif camera == 2 :
                    X = Xc2
                x1, x2, x3 = x
                X1, X2 = X
                Magnification[camera-1] = magnification (X1, X2, x1, x2)
            px = 1/np.mean(Magnification)
            
            #Calcul the retroprojection error
            try :
                z_mean, z_std, z_iter = retroprojection_error('direct',
                                                               direct_pform,
                                                               direct_constants,
                                                               z_points,
                                                               all_X,
                                                               all_xth,
                                                               **kwargs)
            
                mz_points = np.mean(z_points, axis=1)
                z_error = (z_mean-mz_points)*px
                
                plt.errorbar(mz_points, z_error, (z_std)*px, linestyle='None', marker='^')
                # plt.title('Direct retroprojection error for iteration '+str(it+1))
                # plt.xlabel('z theoretical (mm)')
                plt.title("Erreur de rétroprojection (directe) : iteration N°"+str(it+1), size=17)
                plt.xlabel('z théorique (mm)', size=17)
                plt.ylabel('$\Delta$z (px)', size=17)
                plt.savefig(save_retro+'Retroprojection_error_direct_iteration'+str(it+1)+'.pdf', dpi = 500)
                plt.close()
                
                # Save the z list
                np.save(save_retro+"z_list_direct.npy", z_points)
                
                # Change the list to the new ajusted
                z_points = z_iter
            except :
                print('Retroprojection estimation not possible : The ChAruco pattern might be partially out of field of view')
                z_error = np.array([0,0])
                
        # Error of projection
        print ('DEPTH OF FIELD :\n\t The calibrated depth of field is between \n\t', 
               np.min(z_list), 'mm and', np.max(z_list), 'mm.\n')    
        
        print('RETROPROJECTION ERROR : \n\t Max ; min (polynomial form', 
              str(direct_pform),') \n\t =',
              str(sgf.round(np.nanmax(z_error), sigfigs =3)),';',
              str(sgf.round(np.nanmin(z_error), sigfigs =3)),'px\n\t =',
              str(sgf.round(np.nanmax(z_error/px), sigfigs =3)),';',
              str(sgf.round(np.nanmin(z_error/px), sigfigs =3)),'mm')
    
    except :
        z_list = z_recover
        print('No iterations possible')
        # Detect points from folders
        if multifolder :
            all_X, all_xth, nb_pts = data.multifolder_pattern_detection(**kwargs)          
        else :
            all_X, all_xth, nb_pts = data.pattern_detection(**kwargs)
    
        # Creation of the reference matrix Xref and the real position Ucam for 
        # each camera
        nz, npts, _ = all_xth.shape
        z_points = np.ones((nz//2, npts))
        for i in range(nz//2) :
            z_points[i] = z_points[i]*z_list[i]
        
        # Creation of the reference matrix Xref and the real position Ucam for 
        # each camera
        x, Xc1, Xc2 = data.camera_coordinates(all_X, all_xth, z_points)

        # Calcul of the direct polynome's constants. X = A . M
        # Do the system x = Ap*M, where M is the monomial of the real 
        # coordinates of crosses and x the image coordinates, and M the unknow
        M = solvel.direct_Polynome({'polynomial_form' : direct_pform}).pol_form(Xc1, Xc2)
        Ap = np.matmul(x, np.linalg.pinv(M))
        direct_constants = np.asarray(Ap)
        
        # Find and eventually plot the references plans
        if plotting :
            fit, z_error, mean_error, residual = solvel.refplans(x, z_list, plotting = True)
        
        # Compute the magnification (same for each cam as set up is symetric)
        Magnification = np.zeros((2, 2))
        for camera in [1, 2] :
            if camera == 1 :
                X = Xc1
            elif camera == 2 :
                X = Xc2
            x1, x2, x3 = x
            X1, X2 = X
            Magnification[camera-1] = magnification (X1, X2, x1, x2)
        px = 1/np.mean(Magnification)
    
    return(direct_constants, Magnification)    

def Zernike_calibration (z_list : np.ndarray,
                         Zernike_pform : int,
                         multifolder : bool = False,
                         plotting : bool = False,
                         iterations = 1,
                         **kwargs) -> (np.ndarray, 
                                       np.ndarray, 
                                       np.ndarray):
    """Calculation of the magnification between reals and detected positions 
    and the calibration parameters A = Zernike_constants0 (Resp Zernike_constants):--> X = A.M(x)
    
    Args:
       z_list : numpy.ndarray
           List of the different z position. (WARNING : Should be order the 
                                              same way in the target folder)
       Zernike_pform : int
           Polynomial form
           1 = Tilt : 1 + x + y 
           2 add defocus : 2r^2-1 
           3 add astigmatism : x^2-y^2 and 2xy
           4 add coma : (3r^2-2)x and (3r^2-2)y
           5 add trefoil : (3x^2-y^2)y and (3y^2-x^2)x
           6 add sphericity : 6r^4-6r^2+1 
           7 add second astigmatism : 4(x^4-y^4)-3(x^2-y^2) and (8r^2-6)xy
           8 add tetrafoil :  x^4+y^4-6x^2y^2 and 4(x^2-y^2)xy
           9 add second coma :  (10r^4-12r^2+3)x and (10r^4-12r^2+3)y
           10 add second trefoil :  (5x^4-10x^2y^2-15y^4-4x^2+12y^2)x and (15x^4+10x^2y^2-5y^4-12x^2+4y^2)y
           11 add hexafoil :  (x^4-10x^2y^2+5y^4)x and (5x^4-10x^2y^2+y^4)y
           12 add second sphericity :  20r^6-30r^4+12r2-1 
       multifolder : bool, optional
           Used for specific image acquisition when all directions moved
       plotting : Bool
           Plot the calibration view or not
       iterations : int
           Number of iterations. For each of them, the z_list is changed 
           with the retroprojection error assuming that the set-up is not 
           perfect
       **kwargs : All the arguments of the fonction data.pattern_detection
           
    Returns
       Zernike_constants : numpy.ndarray
           Constants of Zernike polynomial form chose (Zernike_pform)
       Magnification : numpy.ndarray
           Magnification between reals and detected positions 
           [[Mag cam1 x, Mag cam1 y], [Mag cam2 x, Mag cam2 y]]
    """
    if not Zernike_pform in list(range(1, 13)) :
        raise('Only define for polynomial forms '+ str(list(range(1, 13))))
    
    z_list = np.array(z_list)
    try :
        save_retro = kwargs['saving_folder']+'/Pycaso_retroprojection_error/'
    except :
        save_retro = 'Pycaso_retroprojection_error'
    if not os.path.exists(save_retro) :
        P = pathlib.Path(save_retro)
        pathlib.Path.mkdir(P, parents = True)
    
    # test if iteration is possible
    z_recover = z_list*1
    try :
        for it in range(iterations) :
            if it == 0 :
                # Detect points from folders
                if multifolder :
                    all_X, all_xth, nb_pts = data.multifolder_pattern_detection(**kwargs)          
                else :
                    all_X, all_xth, nb_pts = data.pattern_detection(**kwargs)        
        
                # Using not well detected images, remove corresponding arrays from all_X 
                # (cam2 and cam1) and z_list
                nz,npts,_ = all_X.shape
                a = np.where(np.isnan(all_X[:,0,0]) == True)[0]
                b, c = [], []
                for i in a :
                    b.append(i)
                    if i < nz//2 :
                        b.append(i+nz//2)
                        c.append(i)
                    else :
                        b.append(i-nz//2)
                        c.append(i-nz//2)
                
                b = list(set(b))
                c = list(set(c))
                b.sort()
                c.sort()
                z_list = np.delete(z_list, c, axis = 0)
                all_xth = np.delete(all_xth, b, axis = 0)
                all_X = np.delete(all_X, b, axis = 0)
                
                # Camera dimensions
                Cameras_dimensions = data.cameras_size(**kwargs)
            
                # Creation of the reference matrix Xref and the real position Ucam for 
                # each camera
                nz, npts, _ = all_xth.shape
                z_points = np.ones((nz//2, npts))
                for i in range(nz//2) :
                    z_points[i] = z_points[i]*z_list[i]
                
            # Creation of the reference matrix Xref and the real position Ucam for 
            # each camera
            x, Xc1, Xc2 = data.camera_coordinates(all_X, all_xth, z_points)
    
            # Calcul of the Zernike polynome's constants. X = A . M
            # Do the system x = Ap*M, where M is the monomial of the real 
            # coordinates of crosses and x the image coordinates, and M the unknow
            M = solvel.Zernike_Polynome({'polynomial_form' : Zernike_pform}).pol_form(Xc1, Xc2, Cameras_dimensions)
            Ap = np.matmul(x, np.linalg.pinv(M))
            Zernike_constants = np.asarray(Ap)
            
            # Find and eventually plot the references plans
            if plotting :
                fit, errors, mean_error, residual = solvel.refplans(x, z_list, plotting = True)
    
            # Compute the magnification (same for each cam as set up is symetric)
            Magnification = np.zeros((2, 2))
            for camera in [1, 2] :
                if camera == 1 :
                    X = Xc1
                elif camera == 2 :
                    X = Xc2
                x1, x2, x3 = x
                X1, X2 = X
                Magnification[camera-1] = magnification (X1, X2, x1, x2)
            px = 1/np.mean(Magnification)
    
            #Calcul the retroprojection error
            try :
                z_mean, z_std, z_iter = retroprojection_error('Zernike',
                                                               Zernike_pform,
                                                               Zernike_constants,
                                                               z_points,
                                                               all_X,
                                                               all_xth,
                                                               **kwargs)
                
    
                mz_points = np.mean(z_points, axis=1)
                z_error = (z_mean-mz_points)*px
                
                plt.errorbar(mz_points, z_error, (z_std)*px, linestyle='None', marker='^')
                # plt.title('Zernike retroprojection error for iteration '+str(it+1))
                # plt.xlabel('z theoretical (mm)')
                plt.title("Erreur de rétroprojection (Zernike) : iteration N°"+str(it+1), size=17)
                plt.xlabel('z théorique (mm)', size=17)
                plt.ylabel('$\Delta$z (px)', size=17)
                plt.savefig(save_retro+'Retroprojection_error_Zernike_iteration'+str(it+1)+'.pdf', dpi = 500)
                plt.close()
                
                # Save the z list
                np.save(save_retro+"z_list_Zernike.npy", z_points)
                
                # Change the list to the new ajusted
                z_points = z_iter
            except :
                print('Retroprojection estimation not possible : The ChAruco pattern might be partially out of field of view')
                z_error = np.array([0,0])
            
        # Error of projection
        print ('DEPTH OF FIELD :\n\t The calibrated depth of field is between \n\t', 
               np.min(z_list), 'mm and', np.max(z_list), 'mm.\n')    
        
        print('RETROPROJECTION ERROR : \n\t Max ; min (polynomial form', 
              str(Zernike_pform),') \n\t =',
              str(sgf.round(np.nanmax(z_error), sigfigs =3)),';',
              str(sgf.round(np.nanmin(z_error), sigfigs =3)),'px\n\t =',
              str(sgf.round(np.nanmax(z_error/px), sigfigs =3)),';',
              str(sgf.round(np.nanmin(z_error/px), sigfigs =3)),'mm')
            
    except :
        z_list = z_recover
        print('No iterations possible')
        # Detect points from folders
        if multifolder :
            all_X, all_xth, nb_pts = data.multifolder_pattern_detection(**kwargs)          
        else :
            all_X, all_xth, nb_pts = data.pattern_detection(**kwargs)
        
        # Camera dimensions
        Cameras_dimensions = data.cameras_size(**kwargs)
    
        # Creation of the reference matrix Xref and the real position Ucam for 
        # each camera
        nz, npts, _ = all_xth.shape
        z_points = np.ones((nz//2, npts))

        for i in range(nz//2) :
            z_points[i] = z_points[i]*z_list[i]
        
        # Creation of the reference matrix Xref and the real position Ucam for 
        # each camera
        x, Xc1, Xc2 = data.camera_coordinates(all_X, all_xth, z_points)

        # Calcul of the Zernike polynome's constants. X = A . M
        # Do the system x = Ap*M, where M is the monomial of the real 
        # coordinates of crosses and x the image coordinates, and M the unknow
        M = solvel.Zernike_Polynome({'polynomial_form' : Zernike_pform}).pol_form(Xc1, Xc2, Cameras_dimensions)
        Ap = np.matmul(x, np.linalg.pinv(M))
        Zernike_constants = np.asarray(Ap)
        
        # Find and eventually plot the references plans
        if plotting :
            fit, z_error, mean_error, residual = solvel.refplans(x, z_list, plotting = True)
        
        # Compute the magnification (same for each cam as set up is symetric)
        Magnification = np.zeros((2, 2))
        for camera in [1, 2] :
            if camera == 1 :
                X = Xc1
            elif camera == 2 :
                X = Xc2
            x1, x2, x3 = x
            X1, X2 = X
            Magnification[camera-1] = magnification (X1, X2, x1, x2)
        px = 1/np.mean(Magnification)
    
    return(Zernike_constants, Magnification)

def Soloff_Zernike_calibration (z_list : np.ndarray,
                                Soloff_pform : int,
                                multifolder : bool = False,
                                plotting : bool = False,
                                iterations = 1,
                                **kwargs) -> (np.ndarray, 
                                              np.ndarray, 
                                              np.ndarray):
            """Calculation of the magnification between reals and detected positions 
            and the calibration parameters A = Soloff_constants0 (Resp Soloff_constants):--> X = A.M(x)
            
            Args:
               z_list : numpy.ndarray
                   List of the different z position. (WARNING : Should be order the 
                                                      same way in the target folder)
               Soloff_pform : int
                   Polynomial form
               multifolder : bool, optional
                   Used for specific image acquisition when all directions moved
               plotting = Bool
                   Plot the calibration view or not
               iterations : int
                   Number of iterations. For each of them, the z_list is changed 
                   with the retroprojection error assuming that the set-up is not 
                   perfect
                **kwargs : All the arguments of the fonction data.pattern_detection
                   
            Returns:
               Soloff_constants0 : numpy.ndarray
                   Constants of Soloff polynomial form '111'
               Soloff_constants : numpy.ndarray
                   Constants of Soloff polynomial form chose (Soloff_pform)
               Magnification : numpy.ndarray
                   Magnification between reals and detected positions 
                   [[Mag cam1 x, Mag cam1 y], [Mag cam2 x, Mag cam2 y]]
            """

            if not Soloff_pform in [111, 1, 221, 222, 2, 332, 333, 3, 443, 444, 4, 554, 555, 5] :
                raise('Only define for polynomial forms 111, 221, 222, 332, 333, 443, 444, 554 or 555')
            
            Soloff_pforms = [1, Soloff_pform]
            
            z_list = np.array(z_list)
            try :
                save_retro = kwargs['saving_folder']+'/Pycaso_retroprojection_error/'
            except :
                save_retro = ''
            if not os.path.exists(save_retro) :
                P = pathlib.Path(save_retro)
                pathlib.Path.mkdir(P, parents = True)
            for it in range(iterations) :
                
                if it == 0 :
                    # Detect points from folders
                    if multifolder :
                        all_X, all_xth, nb_pts = data.multifolder_pattern_detection(**kwargs)          
                    else :
                        all_X, all_xth, nb_pts = data.pattern_detection(**kwargs)        
            
                    # Using not well detected images, remove corresponding arrays from all_X 
                    # (cam2 and cam1) and z_list
                    nz,npts,_ = all_X.shape
                    a = np.where(np.isnan(all_X[:,0,0]) == True)[0]
                    b, c = [], []
                    for i in a :
                        b.append(i)
                        if i < nz//2 :
                            b.append(i+nz//2)
                            c.append(i)
                        else :
                            b.append(i-nz//2)
                            c.append(i-nz//2)
                    
                    b = list(set(b))
                    c = list(set(c))
                    b.sort()
                    c.sort()
                    z_list = np.delete(z_list, c, axis = 0)
                    all_xth = np.delete(all_xth, b, axis = 0)
                    all_X = np.delete(all_X, b, axis = 0)
                    
                    # Camera dimensions
                    Cameras_dimensions = data.cameras_size(**kwargs)
                
                    # Creation of the reference matrix Xref and the real position Ucam for 
                    # each camera
                    nz, npts, _ = all_xth.shape
                    z_points = np.ones((nz//2, npts))
                    for i in range(nz//2) :
                        z_points[i] = z_points[i]*z_list[i]
                
                Soloff_constants0 = []
                Soloff_constants = []
                
                # Creation of the reference matrix Xref and the real position Ucam for 
                # each camera
                x, Xc1, Xc2 = data.camera_coordinates(all_X, all_xth, z_points)     
            
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
                    
                    for pol in range (2) :
                        # Do the system X = Ai*M, where M is the monomial of the real 
                        # coordinates of crosses and X the image coordinates, and M the 
                        # unknow (polynomial form aab)
                        Soloff_pform = Soloff_pforms[pol]
                        M = solvel.Soloff_Polynome({'polynomial_form' : Soloff_pform}).pol_form(x)
                        Ai = np.matmul(X, np.linalg.pinv(M))
                        if pol == 0 :
                            Soloff_constants0.append(Ai)
                        else :
                            Soloff_constants.append(Ai)
                px = 1/np.mean(Magnification)
                    
                # Find and eventually plot the references plans
                if plotting :
                    fit, errors, mean_error, residual = solvel.refplans(x, z_list, plotting = True)
                    
                Soloff_constants0 = np.asarray(Soloff_constants0)
                Soloff_constants = np.asarray(Soloff_constants)
                
                #Calcul the retroprojection error
                try :
                    z_mean, z_std, z_iter = retroprojection_error('Soloff',
                                                                   Soloff_pform,
                                                                   (Soloff_constants0,Soloff_constants),
                                                                   z_points,
                                                                   all_X,
                                                                   all_xth,
                                                                   **kwargs)
                    mz_points = np.mean(z_points, axis=1)
                    z_error = (z_mean-mz_points)*px
                    
                    plt.errorbar(mz_points, z_error, (z_std)*px, linestyle='None', marker='^')
                    # plt.title('Soloff retroprojection error for iteration '+str(it+1))
                    # plt.xlabel('z theoretical (mm)')
                    plt.title("Erreur de rétroprojection (Soloff) pour l'iteration N°"+str(it+1))
                    plt.xlabel('z théorique (mm)')
                    plt.ylabel('$\Delta$z (px)')
                    plt.savefig(save_retro+'Retroprojection_error_Soloff_iteration'+str(it+1)+'.pdf', dpi = 500)
                    plt.close()
                    
                    # Change the list to the new ajusted
                    z_points = z_iter
                except :
                    print('Retroprojection estimation not possible : The ChAruco pattern might be partially out of field of view')
                    z_error = np.array([0,0])
                    
            # Error of projection
            print ('DEPTH OF FIELD :\n\t The calibrated depth of field is between \n\t', 
                   np.min(z_list), 'mm and', np.max(z_list), 'mm.\n')    
            
            print('RETROPROJECTION ERROR : \n\t Max ; min (polynomial form', 
                  str(Soloff_pform),') \n\t =',
                  str(sgf.round(np.nanmax(z_error), sigfigs =3)),';',
                  str(sgf.round(np.nanmin(z_error), sigfigs =3)),'px\n\t =',
                  str(sgf.round(np.nanmax(z_error/px), sigfigs =3)),';',
                  str(sgf.round(np.nanmin(z_error/px), sigfigs =3)),'mm')
                
            return(Soloff_constants0, Soloff_constants, Magnification)

def projector_3D (calibration_dict : dict) -> np.ndarray :
    """Calculation of the projection angle between 2D and 3D axis (x and y)
    
    Args:
       calibration_dict : dict
           Dictionnary of calibration 
           
    Returns:
       projector_3D : int
           projection angle (in degrees) between 2D and 3D axis (x and y)
    """
    name = calibration_dict['name']
    saving_calibration = calibration_dict['saving_folder']
    all_X_file = saving_calibration+'/all_X_' + name + '.npy'
    ncx = calibration_dict['ncx']
    ncy = calibration_dict['ncy']
    all_X = np.load(all_X_file)
    i,j,_ = all_X.shape
    all_X = all_X[i//2:]
    all_X = all_X.reshape(i//2, ncy-1, ncx-1,2)
    all_X = np.nanmean(all_X, axis = 0)
    Gxy,Gxx = np.gradient(all_X[:,:,0])
    Gyy,Gyx = np.gradient(all_X[:,:,1])
    Sin_theta = (np.nanmean(Gxy) - np.nanmean(Gyx))/2
    Cos_theta = (np.nanmean(Gxx) + np.nanmean(Gyy))/2
    size_factor = math.sqrt(Cos_theta*Cos_theta+Sin_theta*Sin_theta)
    Cos_theta = Cos_theta/size_factor
    Sin_theta = Sin_theta/size_factor
    a_acos = math.acos(Cos_theta)
    if Sin_theta < 0:
       projector_3D = math.degrees(-a_acos) % 360
    else: 
       projector_3D = math.degrees(a_acos)
       
    return(projector_3D)

def hybrid_identification(Xc1_identified : np.ndarray,
                          Xc2_identified : np.ndarray,
                          direct_constants : np.ndarray,
                          direct_pform : int,
                          Soloff_constants0 : np.ndarray, 
                          Soloff_constants : np.ndarray,
                          Soloff_pform : int,
                          mask_median : list,
                          method : str = 'curve_fit') -> np.ndarray :
    """Identification of the points detected on both cameras cam1 and cam2 
    into the global 3D-space using direct method and Soloff method when direct
    can't do it well.
    
    Args:
       Xc1_identified : numpy.ndarray
           Points identified on the cam1 camera
       Xc2_identified : numpy.ndarray
           Points identified on the cam2 camera
       direct_constants : numpy.ndarray
           Constants of direct polynomial
       direct_pform : int
           Polynomial form
       Soloff_constants0 : numpy.ndarray
           Constants of Soloff polynomial form '111'
       Soloff_constants : numpy.ndarray
           Constants of Soloff polynomial form chose (polynomial_form)
       Soloff_pform : int
           Polynomial form
       mask_median : list
           Mask used to replace on direct method + the median of the difference
           between Soloff and direct solutions
       method : str, optional
           Python method used to solve it ('Least-squares' or 'curve-fit')
           
    Returns:
       direct_mask : numpy.ndarray
           Identification in the 3D space of the detected points
    """    
    mask, median = mask_median
    xsolution = direct_identification (Xc1_identified,
                                       Xc2_identified,
                                       direct_constants,
                                       direct_pform)
    xsolution = xsolution - median
    mask_crop = np.empty((mask.shape[0],mask.shape[1],2))
    mask_crop[:,:,0], mask_crop[:,:,1] = mask, mask
    
    Xc1_identified_crop = np.ma.MaskedArray(Xc1_identified, 
                                            mask=mask_crop)
    Xc2_identified_crop = np.ma.MaskedArray(Xc2_identified, 
                                            mask=mask_crop)
    
    xSoloff = Soloff_identification (Xc1_identified_crop,
                                     Xc2_identified_crop,
                                     Soloff_constants0, 
                                     Soloff_constants,
                                     Soloff_pform,
                                     method = method)
    
    
    
    for i in range (len(xSoloff[0])) :
        for j in range (len(xSoloff[0,i])) :
            if math.isnan(xSoloff[0, i, j]) :
                ()
            else : 
                xsolution[:, i, j] = xSoloff[:, i, j]
    
    return(xsolution, mask_median)
    
def hybrid_mask (Xc1_identified : np.ndarray,
                 Xc2_identified : np.ndarray,
                 direct_constants : np.ndarray,
                 direct_pform : int,
                 Soloff_constants0 : np.ndarray, 
                 Soloff_constants : np.ndarray,
                 Soloff_pform : int,
                 method : str = 'curve_fit',
                 ROI : bool = False,
                 kernel : int = 5,
                 mask_median : np.ndarray = np.array([False]),
                 gate : int = 5) -> (np.ndarray,
                                     list) :
    """Identification of the points detected on both cameras cam1 and cam2 
    into the global 3D-space using direct and Soloff methods. Detect the
    positions where directe method is not efficient and define the related mask.
    
    Args:
       Xc1_identified : numpy.ndarray
           Points identified on the cam1 camera
       Xc2_identified : numpy.ndarray
           Points identified on the cam2 camera
       direct_constants : numpy.ndarray
           Constants of direct polynomial
       direct_pform : int
           Polynomial form
       Soloff_constants0 : numpy.ndarray
           Constants of Soloff polynomial form '111'
       Soloff_constants : numpy.ndarray
           Constants of Soloff polynomial form chose (polynomial_form)
       Soloff_pform : int
           Polynomial form
       method : str, optional
           Python method used to solve it ('Least-squares' or 'curve-fit')
       ROI : boolean, optional
           Region Of Interest
       kernel : int, optional
           Size of smoothing filter
       mask_median : list
           Mask used to replace on direct method + the median of the difference
           between Soloff and direct solutions
       gate : int, optional
           Output value (in µm) where the mask is True
           
    Returns:
       xsolution : numpy.ndarray
           Solution
       mask_median : list
           Mask used to replace on direct method + the median of the difference
           between Soloff and direct solutions
    """        
    if len(mask_median) == 1 :
        xdirect = direct_identification (Xc1_identified,
                                         Xc2_identified,
                                         direct_constants,
                                         direct_pform)

        xSoloff = Soloff_identification (Xc1_identified,
                                         Xc2_identified,
                                         Soloff_constants0, 
                                         Soloff_constants,
                                         Soloff_pform,
                                         method = method)
        
        image = xdirect[2] - xSoloff[2]
        
        mask_median = data.hybrid_mask_creation(image,
                                                ROI = ROI,
                                                kernel = kernel,
                                                gate = gate)
    
    xsolution, mask_median = hybrid_identification(Xc1_identified,
                                                   Xc2_identified,
                                                   direct_constants,
                                                   direct_pform,
                                                   Soloff_constants0, 
                                                   Soloff_constants,
                                                   Soloff_pform,
                                                   mask_median,
                                                   method = method)

    return(xsolution, mask_median)
    
def AI_training (X_c1 : np.ndarray,
                 X_c2 : np.ndarray,
                 xSoloff_solution : np.ndarray,
                 AI_training_size : int = 50000,
                 file : str = 'Soloff_AI_training.csv',
                 method : str = 'simultaneously') -> sklearn.ensemble._forest.RandomForestRegressor :
    """Training the AI metamodel with already known datas.
    
    Args:
       X_c1 : numpy.ndarray
           cam1 coordinates of points.
       X_c2 : numpy.ndarray
           cam2 coordinates of points.
       xSoloff_solution : numpy.ndarray
           3D space coordinates identified with Soloff method.
       AI_training_size : int
           Number of datas (points) used to train the AI metamodel.
       file : str
           Name of saving file for training
       method : str
           AI method :
                   - Simultaneously = x,y and z in the same model
                   - Independantly = x,y and z in different models
                   
    Returns:
       model : sklearn.ensemble._forest.RandomForestRegressor
           AI model or list of AI models
    """
    # Organise the list of parameters
    if len(X_c1.shape) == 3 :
        Xc10, Yc10 = np.ravel(X_c1[:,:,0]), np.ravel(X_c1[:,:,1])
        Xc20, Yc20 = np.ravel(X_c2[:,:,0]), np.ravel(X_c2[:,:,1])
    elif len(X_c1.shape) == 2 :
        Xc10, Yc10 = np.ravel(X_c1[:,0]), np.ravel(X_c1[:,1])
        Xc20, Yc20 = np.ravel(X_c2[:,0]), np.ravel(X_c2[:,1])
    xS, yS, zS = xSoloff_solution
    xS = np.ravel(xS)
    yS = np.ravel(yS)
    zS = np.ravel(zS)
    N = Xc10.shape[0]
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
        spamwriter.writerow(['Xc1'] + ['Yc1'] + ['Xc2'] + ['Yc2'] + 
                            ['x'] + ['y'] + ['z'])
        for i in range (AI_training_size) :
            spamwriter.writerow([str(Xc10[rd_list[i]]), 
                                 str(Yc10[rd_list[i]]), 
                                 str(Xc20[rd_list[i]]), 
                                 str(Yc20[rd_list[i]]), 
                                 str(xS[rd_list[i]]), 
                                 str(yS[rd_list[i]]), 
                                 str(zS[rd_list[i]])])
    # Build the AI model with the AI_training_size.
    if method == 'simultaneously' :
        model, accuracy = solvel.AI_solve_simultaneously (file)
    elif method == 'independantly' :
        modelx, modely, modelz, _, _, _ = solvel.AI_solve_independantly (file)
        model = [modelx, modely, modelz]
    elif method == 'z_dependantly' :
        modelx, modely, modelz, _, _, _ = solvel.AI_solve_independantly (file)
        model = [modelx, modely, modelz]
    else :
        print('No method ', method)
    return(model)
    
def AI_identification (X_c1 : np.ndarray,
                       X_c2 : np.ndarray,
                       model : sklearn.ensemble._forest.RandomForestRegressor,
                       method : str = 'simultaneously') -> np.ndarray:
    """Calculation of the 3D points with AI model.
    
    Args:
       X_c1 : numpy.ndarray
           cam1 coordinates of points.
       X_c2 : numpy.ndarray
           cam2 coordinates of points.
       model : sklearn.ensemble._forest.RandomForestRegressor
           AI model
       method : str
           AI method :
                   - Simultaneously = x,y and z in the same model
                   - Independantly = x,y and z in different models

    Returns:
       xAI_solution : numpy.ndarray
           3D space coordinates identified with AI method.
    """
    # Solve on all pts
    if len(X_c1.shape) == 3 :
        Xc10, Yc10 = np.ravel(X_c1[:,:,0]), np.ravel(X_c1[:,:,1])
        Xc20, Yc20 = np.ravel(X_c2[:,:,0]), np.ravel(X_c2[:,:,1])
    elif len(X_c1.shape) == 2 :
        Xc10, Yc10 = np.ravel(X_c1[:,0]), np.ravel(X_c1[:,1])
        Xc20, Yc20 = np.ravel(X_c2[:,0]), np.ravel(X_c2[:,1])
    X = np.transpose(np.array([Xc10, Yc10, 
                               Xc20, Yc20]))   
     
    if method == 'simultaneously' :
        xAI_solution = model.predict(X)
        xAI_solution = np.transpose(xAI_solution)
    elif method == 'independantly' or method == 'z_dependantly' :
        modelx, modely, modelz = model
        xAI_solutionx = modelx.predict(X)
        xAI_solutiony = modely.predict(X)
        xAI_solutionz = modelz.predict(X)
        xAI_solution = np.array([xAI_solutionx, xAI_solutiony, xAI_solutionz])
    else :
        print('No method ', method)
    if len(X_c1.shape) == 3 :
        nx, ny, __ = X_c1.shape
        xAI_solution = xAI_solution.reshape(3, nx, ny)
    return (xAI_solution)
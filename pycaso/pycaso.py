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
    """Identification of the points detected on both cameras left and right 
    into the global 3D-space
    
    Args:
       Xc1_identified : numpy.ndarray
           Points identified on the left camera
       Xc2_identified : numpy.ndarray
           Points identified on the right camera
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
        Xl = np.copy(Xc1_identified[:,:,0])
        Yl = np.copy(Xc1_identified[:,:,1])
        Xr = np.copy(Xc2_identified[:,:,0])
        Yr = np.copy(Xc2_identified[:,:,1])
        Xl0 = []
        Yl0 = []
        Xr0 = []
        Yr0 = []

        l = 20
        nx,ny = Xl.shape
        nx = nx//l
        ny = ny//l
        for i in range (l) :
            for j in range (l) :
                Xl0.append(Xl[i*nx, j*ny])
                Yl0.append(Yl[i*nx, j*ny])
                Xr0.append(Xr[i*nx, j*ny])
                Yr0.append(Yr[i*nx, j*ny])
        
        Xl0 = np.array(Xl0).reshape((l,l))
        Yl0 = np.array(Yl0).reshape((l,l))
        Xr0 = np.array(Xr0).reshape((l,l))
        Yr0 = np.array(Yr0).reshape((l,l))
        Xc1_identified0 = np.zeros((l,l,2))
        Xc1_identified0[:,:,0] = Xl0
        Xc1_identified0[:,:,1] = Yl0
        Xc2_identified0 = np.zeros((l,l,2))
        Xc2_identified0[:,:,0] = Xr0
        Xc2_identified0[:,:,1] = Yr0        
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
        xsolution = solvel.Peter(Xl,
                                 Yl,
                                 Xr,
                                 Yr,
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
    """Identification of the points detected on both cameras left and right 
    into the global 3D-space
    
    Args:
       Xc1_identified : numpy.ndarray
           Points identified on the left camera
       Xc2_identified : numpy.ndarray
           Points identified on the right camera
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
    Xl1, Xl2 = Xc1_identified[:,0], Xc1_identified[:,1]
    Xr1, Xr2 = Xc2_identified[:,0], Xc2_identified[:,1]
    Xl = np.zeros((2,len(Xl1)))
    Xr = np.zeros((2,len(Xr1)))
    Xl = Xl1, Xl2
    Xr = Xr1, Xr2
    
    M = solvel.Direct_Polynome({'polynomial_form' : direct_pform}).pol_form(Xl, Xr)
    xsolution = np.matmul(direct_constants,M)
    if modif_22_12_09 :
        xsolution = xsolution.reshape((3, nx, ny))
    return(xsolution)

def Zernike_identification (Xc1_identified : np.ndarray,
                            Xc2_identified : np.ndarray,
                            Zernike_constants : np.ndarray,
                            Zernike_pform : int,
                            Cameras_dimensions : list) -> np.ndarray :
    """Identification of the points detected on both cameras left and right 
    into the global 3D-space
    
    Args:
       Xc1_identified : numpy.ndarray
           Points identified on the left camera
       Xc2_identified : numpy.ndarray
           Points identified on the right camera
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
    Xl1, Xl2 = Xc1_identified[:,0], Xc1_identified[:,1]
    Xr1, Xr2 = Xc2_identified[:,0], Xc2_identified[:,1]
    Xl = np.zeros((2,len(Xl1)))
    Xr = np.zeros((2,len(Xr1)))
    Xl = Xl1, Xl2
    Xr = Xr1, Xr2
    
    M = solvel.Zernike_Polynome({'polynomial_form' : Zernike_pform}).pol_form(Xl, Xr, Cameras_dimensions)
    xsolution = np.matmul(Zernike_constants,M)
    if modif_22_12_09 :
        xsolution = xsolution.reshape((3, nx, ny))
    return(xsolution)

def retroprojection_error (calibration_method : str,
                           pform : int,
                           calibration_constants : np.ndarray,
                           z_list : np.ndarray,
                           left_folder : str = 'left_calibration',
                           right_folder : str = 'right_calibration',
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
       z_list : numpy.ndarray
           List of the different z position
       left_folder : str, optional
           Left calibration images folder
       right_folder : str, optional
           Right calibration images folder
       name : str, optional
           Name to save
       saving_folder : str, optional
           Folder to save
       ncx : int, optional
           The number of squares for the chessboard through x direction
       ncy : int, optional
           The number of squares for the chessboard through y direction
       method : str, optional
           Python method used to solve Soloff ('Peter' 'curve-fit')
       Other arguments are useless...
           
    Returns:
       z_theoric : list
           z_list without the non detected plans ejected
       z_retro : list
           z position retrocalculated
       z_std : list
           standard deviation of each z_retro
       z_iter : list
           z_retro + the z position not detected of z_list
    """
    Save_all_X = str(saving_folder) + "/all_X_" + name + ".npy"
    Cameras_dimensions = data.cameras_size(left_folder,
                                           right_folder)
    all_X = np.load(Save_all_X)
    z_retro = []
    z_std = []
    i_s=[]
    z_iter = []
    n,npts,_ = all_X.shape
    n = n//2
    for i in range (n) :
        Xc1 = all_X[i]
        Xc2 = all_X[i+n]
        if Xc1.shape == (npts, 2) and Xc2.shape == (npts, 2) :
            nx, ny = ncx-1, ncy-1
            Xc1 = np.reshape(Xc1,(ny, nx, 2))
            Xc2 = np.reshape(Xc2,(ny, nx, 2))
            for l in range (eject_crown) :
                Xc1 = Xc1[1:-1,1:-1,:]
                Xc2 = Xc2[1:-1,1:-1,:]
            ny,nx,_ = Xc1.shape
            if calibration_method == 'Zernike' :
                Zernike_constants = calibration_constants
                Zernike_pform = pform
                Solution = Zernike_identification (Xc1,
                                                   Xc2,
                                                   Zernike_constants,
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
            z_retro.append(np.mean(Solution[2]))
            z_std.append(np.std(Solution[2]))
            z_iter.append(np.mean(Solution[2]))
            
            # fig = plt.figure()
            # im = plt.imshow(np.reshape(Solution[2], (ny,nx)), plt.get_cmap('hot'), alpha=1)
            # cb = plt.colorbar()
            # cb.set_label('z (mm)')
            # plt.title('z distribution on plan = ' +str(i)+ ' ; iteration = ' +str(it) + ' ; method = ' + names[na])
            # plt.savefig(saving_folder + name + '/iteration' +str(it)+ '_z_plan' +str(i)+'.png', dpi = 500)
            # plt.close()
        else :
            i_s.append(i)
            z_iter.append(z_list[i])
            
    z_theoric = np.delete(z_list, i_s)
    argsort = np.argsort(z_theoric)
    z_theoric = z_theoric[argsort]
    z_iter = np.asarray(z_iter)
    z_retro = np.asarray(z_retro)[argsort]
    z_std = np.asarray(z_std)[argsort]
    return(z_theoric, z_retro, z_std, z_iter)

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
           [[Mag Left x, Mag Left y], [Mag Right x, Mag Right y]]
    """

    if not Soloff_pform in [111, 1, 221, 222, 2, 332, 333, 3, 443, 444, 4, 554, 555, 5] :
        raise('Only define for polynomial forms 111, 221, 222, 332, 333, 443, 444, 554 or 555')
    
    Soloff_pforms = [1, Soloff_pform]

    z_list = np.array(z_list)    
    z_list2 = z_list
    try :
        save_retro = kwargs['saving_folder']+'/Pycaso_retroprojection_error/'
    except :
        save_retro = ''
    if not os.path.exists(save_retro) :
        P = pathlib.Path(save_retro)
        pathlib.Path.mkdir(P, parents = True)
    for it in range(iterations) :
        z_listd = z_list2
        Soloff_constants0 = []
        Soloff_constants = []
        # Detect points from folders
        if multifolder :
            all_X, all_x, nb_pts = data.multifolder_pattern_detection(**kwargs)          
        else :
            all_X, all_x, nb_pts = data.pattern_detection(**kwargs)        
    
        # Using not well detected images, remove corresponding arrays from all_X 
        # (right and left) and z_list
        e = 0 ; i = 0
        while e==0 :
            nx,ny,nz = all_X.shape
            if i >= nx :
                e = 1
            else :
                if np.isnan(all_X[i]).all() :
                    if i < nx//2 :
                        ip = i
                    else :
                        ip = i - nx//2
                    all_X = np.delete(all_X,nx//2 + ip, axis = 0)
                    all_X = np.delete(all_X, ip, axis = 0)
                    all_x = np.delete(all_x,nx//2 + ip, axis = 0)
                    all_x = np.delete(all_x, ip, axis = 0)
                    z_listd = np.delete(z_listd, ip, axis = 0)
        
                else : 
                    i+=1
        print ('')
        print ('DEPTH OF FIELD :')
        print ('          The calibrated depth of field is between ', 
               np.min(z_listd), 
               'mm and ', 
               np.max(z_listd), 
               'mm.')    
        print ('')
        
        # Creation of the reference matrix Xref and the real position Ucam for 
        # each camera
        x, Xc1, Xc2 = data.camera_np_coordinates(all_X, all_x, z_listd)     
    
        # Plot the references plans
        if plotting :
            solvel.refplans(x, z_listd, plotting = True)
    
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
        Soloff_constants0 = np.asarray(Soloff_constants0)
        Soloff_constants = np.asarray(Soloff_constants)
        z_theoric, z_retro, z_std, z_iter = retroprojection_error ('Soloff',
                                                                   Soloff_pform,
                                                                   (Soloff_constants0,Soloff_constants),
                                                                   z_list2,
                                                                   **kwargs)
        z_list2 = z_iter
        
        px = 1/np.mean(Magnification)
        z_error = (z_retro-z_theoric)*px
        
        plt.errorbar(z_theoric, z_error, (z_std)*px, linestyle='None', marker='^')
        plt.title('zmeasured - ztheoretical for iteration '+str(it+1))
        plt.xlabel('z theoretical (mm)')
        plt.ylabel('DELTA_z (px)')
        plt.legend()
        plt.savefig(save_retro+'Soloff_retroprojection_error_iteration'+str(it+1)+'.png', dpi = 500)
        plt.close()
        
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
    if not direct_pform in [1, 2, 3, 4, 5] :
        raise ('Only define for polynomial degrees (1, 2, 3, 4 or 5')
        
    z_list = np.array(z_list)
    z_list2 = z_list
    try :
        save_retro = kwargs['saving_folder']+'/Pycaso_retroprojection_error/'
    except :
        save_retro = ''
    if not os.path.exists(save_retro) :
        P = pathlib.Path(save_retro)
        pathlib.Path.mkdir(P, parents = True)
    for it in range(iterations) :
        z_listd = z_list2
        # Detect points from folders
        if multifolder :
            all_X, all_x, nb_pts = data.multifolder_pattern_detection(**kwargs)          
        else :
            all_X, all_x, nb_pts = data.pattern_detection(**kwargs)       
        # Remove nan arrays
        e = 0 ; i = 0
        while e==0 :
            nx,ny,nz = all_X.shape
            if i >= nx :
                e = 1
            else :
                if np.isnan(all_X[i]).all() :
                    if i < nx//2 :
                        ip = i
                    else :
                        ip = i - nx//2
                    all_X = np.delete(all_X,nx//2 + ip, axis = 0)
                    all_X = np.delete(all_X, ip, axis = 0)
                    all_x = np.delete(all_x,nx//2 + ip, axis = 0)
                    all_x = np.delete(all_x, ip, axis = 0)
                    z_listd = np.delete(z_listd, ip, axis = 0)
        
                else : 
                    i+=1
    
        print ('')
        print ('DEPTH OF FIELD :')
        print ('          The calibrated depth of field is between ', np.min(z_listd), 'mm and ', np.max(z_listd), 'mm.')    
        print ('')
    
        # Creation of the reference matrix Xref and the real position Ucam for 
        # each camera i
        x, Xc1, Xc2 = data.camera_np_coordinates(all_X, all_x, z_listd)
    
        # Plot the references plans
        if plotting :
            solvel.refplans(x, z_listd, plotting = True)
            
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
        direct_constants = Ap
        z_theoric, z_retro, z_std, z_iter = retroprojection_error ('direct',
                                                                   direct_pform,
                                                                   direct_constants,
                                                                   z_list2,
                                                                   **kwargs)
        z_list2 = z_iter
    
        # Error of projection
        xd = np.matmul(Ap,M)
        proj_error = x - xd
        print('Max ; min projection error (polynomial form ', 
            str(direct_pform),
            ') for camera ', 
            str(camera),
            ' = ',
            str(sgf.round(np.nanmax(proj_error), sigfigs =3)),
            ' ; ',
            str(sgf.round(np.nanmin(proj_error), sigfigs =3)),
            ' px')
        px = 1/np.mean(Magnification)
        z_error = (z_retro-z_theoric)*px
        
        plt.errorbar(z_theoric, z_error, (z_std)*px, linestyle='None', marker='^')
        plt.title('zmeasured - ztheoretical for iteration '+str(it+1))
        plt.xlabel('z theoretical (mm)')
        plt.ylabel('DELTA_z (px)')
        plt.legend()
        plt.savefig(save_retro+'direct_retroprojection_error_iteration'+str(it+1)+'.png', dpi = 500)
        plt.close()
        

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
           [[Mag Left x, Mag Left y], [Mag Right x, Mag Right y]]
    """
    if not Zernike_pform in list(range(1, 13)) :
        raise('Only define for polynomial forms '+ str(list(range(1, 13))))
    
    z_list = np.array(z_list)  
    z_list2 = z_list
    try :
        save_retro = kwargs['saving_folder']+'/Pycaso_retroprojection_error/'
    except :
        save_retro = ''
    if not os.path.exists(save_retro) :
        P = pathlib.Path(save_retro)
        pathlib.Path.mkdir(P, parents = True)
    for it in range(iterations) :
        z_listd = z_list2
        # Detect points from folders
        if multifolder :
            all_X, all_x, nb_pts = data.multifolder_pattern_detection(**kwargs)          
        else :
            all_X, all_x, nb_pts = data.pattern_detection(**kwargs)        

        # Using not well detected images, remove corresponding arrays from all_X 
        # (right and left) and z_listd
        e = 0 ; i = 0
        while e==0 :
            nx,ny,nz = all_X.shape
            if i >= nx :
                e = 1
            else :
                if np.isnan(all_X[i]).all() :
                    if i < nx//2 :
                        ip = i
                    else :
                        ip = i - nx//2
                    all_X = np.delete(all_X,nx//2 + ip, axis = 0)
                    all_X = np.delete(all_X, ip, axis = 0)
                    all_x = np.delete(all_x,nx//2 + ip, axis = 0)
                    all_x = np.delete(all_x, ip, axis = 0)
                    z_listd = np.delete(z_listd, ip, axis = 0)
        
                else : 
                    i+=1
        print ('')
        print ('DEPTH OF FIELD :')
        print ('          The calibrated depth of field is between ', 
               np.min(z_listd), 
               'mm and ', 
               np.max(z_listd), 
               'mm.')    
        print ('')
        
        # Creation of the reference matrix Xref and the real position Ucam for 
        # each camera
        x, Xc1, Xc2 = data.camera_np_coordinates(all_X, all_x, z_listd)     
        
        # Camera dimensions
        Cameras_dimensions = data.cameras_size(**kwargs)
        
        # Plot the references plans
        if plotting :
            solvel.refplans(x, z_listd, plotting = True)
    
        # Calcul of the Zernike polynome's constants. X = A . M
            
        # Do the system x = Ap*M, where M is the monomial of the real 
        # coordinates of crosses and x the image coordinates, and M the unknow
        M = solvel.Zernike_Polynome({'polynomial_form' : Zernike_pform}).pol_form(Xc1, Xc2, Cameras_dimensions)
        Ap = np.matmul(x, np.linalg.pinv(M))
        Zernike_constants = Ap
        Zernike_constants = np.asarray(Zernike_constants)
        z_theoric, z_retro, z_std, z_iter = retroprojection_error ('Zernike',
                                                                   Zernike_pform,
                                                                   Zernike_constants,
                                                                   z_list2,
                                                                   **kwargs)
        z_list2 = z_iter
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
        # Error of projection
        xd = np.matmul(Ap,M)
        proj_error = x - xd
        print('Max ; min projection error (polynomial form ', 
            str(Zernike_pform),
            ') for camera ', 
            str(camera),
            ' = ',
            str(sgf.round(np.nanmax(proj_error), sigfigs =3)),
            ' ; ',
            str(sgf.round(np.nanmin(proj_error), sigfigs =3)),
            ' px')
        px = 1/np.mean(Magnification)
        z_error = (z_retro-z_theoric)*px
        
        plt.errorbar(z_theoric, z_error, (z_std)*px, linestyle='None', marker='^')
        plt.title('zmeasured - ztheoretical for iteration '+str(it+1))
        plt.xlabel('z theoretical (mm)')
        plt.ylabel('DELTA_z (px)')
        plt.legend()
        plt.savefig(save_retro+'Zernike_retroprojection_error_iteration'+str(it+1)+'.png', dpi = 500)
        plt.close()
        
    return(Zernike_constants, Magnification)

def Soloff_Zernike_calibration (z_list : np.ndarray,
                                Soloff_pform : int,
                                Zernike_pform : int = 3,
                                multifolder : bool = False,
                                plotting : bool = False,
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
       **kwargs : All the arguments of the fonction data.pattern_detection
           
    Returns:
       Soloff_constants0 : numpy.ndarray
           Constants of Soloff polynomial form '111'
       Soloff_constants : numpy.ndarray
           Constants of Soloff polynomial form chose (Soloff_pform)
       Magnification : numpy.ndarray
           Magnification between reals and detected positions 
           [[Mag Left x, Mag Left y], [Mag Right x, Mag Right y]]
    """
    Zernike_constants, Magnification = Zernike_calibration (z_list,
                                                            Zernike_pform ,
                                                            multifolder = multifolder,
                                                            plotting = plotting,
                                                            **kwargs)
    Soloff_constants0, Soloff_constants, Magnification = Soloff_calibration (z_list,
                                                                             Soloff_pform,
                                                                             multifolder = multifolder,
                                                                             plotting = plotting,
                                                                             **kwargs)
    return(Zernike_constants, Soloff_constants, Magnification)

def Soloff_Zernike_identification (Xc1_identified : np.ndarray,
                                   Xc2_identified : np.ndarray,
                                   Zernike_constants : np.ndarray, 
                                   Soloff_constants : np.ndarray,
                                   Soloff_pform : int,
                                   Cameras_dimensions : list,
                                   Zernike_pform : int = 3,
                                   method : str = 'Peter',
                                   cut : int = 0) -> np.ndarray :
    """Identification of the points detected on both cameras left and right 
    into the global 3D-space
    
    Args:
       Xc1_identified : numpy.ndarray
           Points identified on the left camera
       Xc2_identified : numpy.ndarray
           Points identified on the right camera
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
        Xl = np.copy(Xc1_identified[:,:,0])
        Yl = np.copy(Xc1_identified[:,:,1])
        Xr = np.copy(Xc2_identified[:,:,0])
        Yr = np.copy(Xc2_identified[:,:,1])
        Xl0 = []
        Yl0 = []
        Xr0 = []
        Yr0 = []

        l = 20
        nx,ny = Xl.shape
        nx = nx//l
        ny = ny//l
        for i in range (l) :
            for j in range (l) :
                Xl0.append(Xl[i*nx, j*ny])
                Yl0.append(Yl[i*nx, j*ny])
                Xr0.append(Xr[i*nx, j*ny])
                Yr0.append(Yr[i*nx, j*ny])
        
        Xl0 = np.array(Xl0).reshape((l,l))
        Yl0 = np.array(Yl0).reshape((l,l))
        Xr0 = np.array(Xr0).reshape((l,l))
        Yr0 = np.array(Yr0).reshape((l,l))
        Xc1_identified0 = np.zeros((l,l,2))
        Xc1_identified0[:,:,0] = Xl0
        Xc1_identified0[:,:,1] = Yl0
        Xc2_identified0 = np.zeros((l,l,2))
        Xc2_identified0[:,:,0] = Xr0
        Xc2_identified0[:,:,1] = Yr0        
        x,y,z = Zernike_identification (Xc1_identified0,
                                        Xc2_identified0,
                                        Zernike_constants = Zernike_constants,
                                        Zernike_pform = Zernike_pform,
                                        Cameras_dimensions = Cameras_dimensions)
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
        xsolution = solvel.Peter(Xl,
                                 Yl,
                                 Xr,
                                 Yr,
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
        x0 = Zernike_identification (Xc1_identified,
                                     Xc2_identified,
                                     Zernike_constants = Zernike_constants,
                                     Zernike_pform = Zernike_pform,
                                     Cameras_dimensions = Cameras_dimensions)
        
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




def hybrid_identification(Xc1_identified : np.ndarray,
                          Xc2_identified : np.ndarray,
                          direct_constants : np.ndarray,
                          direct_pform : int,
                          Soloff_constants0 : np.ndarray, 
                          Soloff_constants : np.ndarray,
                          Soloff_pform : int,
                          mask_median : list,
                          method : str = 'curve_fit') -> np.ndarray :
    """Identification of the points detected on both cameras left and right 
    into the global 3D-space using direct method and Soloff method when direct
    can't do it well.
    
    Args:
       Xc1_identified : numpy.ndarray
           Points identified on the left camera
       Xc2_identified : numpy.ndarray
           Points identified on the right camera
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
    """Identification of the points detected on both cameras left and right 
    into the global 3D-space using direct and Soloff methods. Detect the
    positions where directe method is not efficient and define the related mask.
    
    Args:
       Xc1_identified : numpy.ndarray
           Points identified on the left camera
       Xc2_identified : numpy.ndarray
           Points identified on the right camera
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
                 AI_training_size : int = 1000,
                 file : str = 'Soloff_AI_training.csv',
                 method : str = 'simultaneously') -> sklearn.ensemble._forest.RandomForestRegressor :
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
        Xl0, Yl0 = np.ravel(X_c1[:,:,0]), np.ravel(X_c1[:,:,1])
        Xr0, Yr0 = np.ravel(X_c2[:,:,0]), np.ravel(X_c2[:,:,1])
    elif len(X_c1.shape) == 2 :
        Xl0, Yl0 = np.ravel(X_c1[:,0]), np.ravel(X_c1[:,1])
        Xr0, Yr0 = np.ravel(X_c2[:,0]), np.ravel(X_c2[:,1])
    xS, yS, zS = xSoloff_solution
    xS = np.ravel(xS)
    yS = np.ravel(yS)
    zS = np.ravel(zS)
    N = Xl0.shape[0]
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
           Left coordinates of points.
       X_c2 : numpy.ndarray
           Right coordinates of points.
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
        Xl0, Yl0 = np.ravel(X_c1[:,:,0]), np.ravel(X_c1[:,:,1])
        Xr0, Yr0 = np.ravel(X_c2[:,:,0]), np.ravel(X_c2[:,:,1])
    elif len(X_c1.shape) == 2 :
        Xl0, Yl0 = np.ravel(X_c1[:,0]), np.ravel(X_c1[:,1])
        Xr0, Yr0 = np.ravel(X_c2[:,0]), np.ravel(X_c2[:,1])
    X = np.transpose(np.array([Xl0, Yl0, 
                               Xr0, Yr0]))   
     
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
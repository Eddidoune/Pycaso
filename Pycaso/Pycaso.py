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

def magnification (X1 : np.ndarray, 
                   X2 : np.ndarray,
                   x1 : np.ndarray,
                   x2 : np.ndarray) -> np.ndarray :
    """Calculation of the magnification between reals and detected positions
    in mm/px
    
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

def Soloff_calibration (__calibration_dict__ : dict,
                        z_list : np.ndarray, 
                        Soloff_pform : int,
                        hybrid_verification : bool = False ,
                        multifolder : bool = False,
                        plotting : bool = False) -> (np.ndarray, 
                                                     np.ndarray, 
                                                     np.ndarray):
    """Calculation of the magnification between reals and detected positions 
    and the calibration parameters A = Soloff_constants0 (Resp Soloff_constants):--> X = A.M(x)
    
    Args:
       __calibration_dict__ : dict
           Calibration properties define in a dict. Including 'left_folder', 
           'right_folder', 'name', 'ncx', 'ncy', 'sqr'
       z_list : numpy.ndarray
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
       multifolder : bool, optional
           Used for specific image acquisition when all directions moved
       plotting = Bool
           Plot the calibration view or not
           
    Returns:
       Soloff_constants0 : numpy.ndarray
           Constants of Soloff polynomial form '111'
       Soloff_constants : numpy.ndarray
           Constants of Soloff polynomial form chose (Soloff_pforloasm)
       Magnification : numpy.ndarray
           Magnification between reals and detected positions 
           [[Mag Left x, Mag Left y], [Mag Right x, Mag Right y]]
    """
    z_list = np.array(z_list)    
    Soloff_constants0 = np.zeros((2, 2, 4))
    if Soloff_pform == 111 or Soloff_pform == 1 :
        Soloff_constants = np.zeros((2, 2, 4))
    elif Soloff_pform == 221 :
        Soloff_constants = np.zeros((2, 2, 9))
    elif Soloff_pform == 222 or Soloff_pform == 2 :
        Soloff_constants = np.zeros((2, 2, 10))
    elif Soloff_pform == 332 :
        Soloff_constants = np.zeros((2, 2, 19))
    elif Soloff_pform == 333 or Soloff_pform == 3 :
        Soloff_constants = np.zeros((2, 2, 20))
    elif Soloff_pform == 443 :
        Soloff_constants = np.zeros((2, 2, 34))
    elif Soloff_pform == 444 or Soloff_pform == 4 :
        Soloff_constants = np.zeros((2, 2, 35))
    elif Soloff_pform == 554 :
        Soloff_constants = np.zeros((2, 2, 55))
    elif Soloff_pform == 555 or Soloff_pform == 5 :
        Soloff_constants = np.zeros((2, 2, 56))    
    else :
        raise ('Only define for polynomial forms 111, 221, 222, 332, 333, 443, 444, 554 or 555')
    
    A_0 = [Soloff_constants0, Soloff_constants]
    Soloff_pforms = [1, Soloff_pform]

    
    # Detect points from folders
    if multifolder :
        all_X, all_x, nb_pts = data.multifolder_pattern_detection(__calibration_dict__,
                                                                  hybrid_verification = hybrid_verification)          
    else :
        all_X, all_x, nb_pts = data.pattern_detection(__calibration_dict__,
                                                      hybrid_verification = hybrid_verification)        

    # Creation of the reference matrix Xref and the real position Ucam for 
    # each camera
    x, Xc1, Xc2 = data.camera_np_coordinates(all_X, all_x, z_list)     

    # Plot the references plans
    solvel.refplans(x, z_list, plotting = plotting)

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
    Soloff_constants0, Soloff_constants = A_0
    return(Soloff_constants0, Soloff_constants, Magnification)

def Soloff_calibration2 (z_list : np.ndarray,
                         Soloff_pform : int,
                         left_folder : str = 'left_calibration',
                         right_folder : str = 'right_calibration',
                         name : str = 'calibration',
                         saving_folder : str = 'results',
                         ncx : int = 16,
                         ncy : int = 12,
                         sqr : float = 0.3,
                         hybrid_verification : bool = False,
                         multifolder : bool = False,
                         plotting : bool = False) -> (np.ndarray, 
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
       sqr : float, optional
           Size of a square (in mm)
       hybrid_verification : bool, optional
           If True, verify each pattern detection and propose to pick 
           manually the bad detected corners. The image with all detected
           corners is show and you can decide to change any point using
           it ID (ID indicated on the image) as an input. If there is no
           bad detected corner, press ENTER to go to the next image.
       multifolder : bool, optional
           Used for specific image acquisition when all directions moved
       plotting = Bool
           Plot the calibration view or not
           
    Returns:
       Soloff_constants0 : numpy.ndarray
           Constants of Soloff polynomial form '111'
       Soloff_constants : numpy.ndarray
           Constants of Soloff polynomial form chose (Soloff_pforloasm)
       Magnification : numpy.ndarray
           Magnification between reals and detected positions 
           [[Mag Left x, Mag Left y], [Mag Right x, Mag Right y]]
    """
    z_list = np.array(z_list)    
    Soloff_constants0 = np.zeros((2, 2, 4))
    if Soloff_pform == 111 or Soloff_pform == 1 :
        Soloff_constants = np.zeros((2, 2, 4))
    elif Soloff_pform == 221 :
        Soloff_constants = np.zeros((2, 2, 9))
    elif Soloff_pform == 222 or Soloff_pform == 2 :
        Soloff_constants = np.zeros((2, 2, 10))
    elif Soloff_pform == 332 :
        Soloff_constants = np.zeros((2, 2, 19))
    elif Soloff_pform == 333 or Soloff_pform == 3 :
        Soloff_constants = np.zeros((2, 2, 20))
    elif Soloff_pform == 443 :
        Soloff_constants = np.zeros((2, 2, 34))
    elif Soloff_pform == 444 or Soloff_pform == 4 :
        Soloff_constants = np.zeros((2, 2, 35))
    elif Soloff_pform == 554 :
        Soloff_constants = np.zeros((2, 2, 55))
    elif Soloff_pform == 555 or Soloff_pform == 5 :
        Soloff_constants = np.zeros((2, 2, 56))    
    else :
        raise('Only define for polynomial forms 111, 221, 222, 332, 333, 443, 444, 554 or 555')
    
    A_0 = [Soloff_constants0, Soloff_constants]
    Soloff_pforms = [1, Soloff_pform]

    
    # Detect points from folders
    if multifolder :
        all_X, all_x, nb_pts = data.multifolder_pattern_detection(left_folder = left_folder,
                                                                  right_folder = right_folder,
                                                                  name = name,
                                                                  saving_folder = saving_folder,
                                                                  ncx = ncx,
                                                                  ncy = ncy,
                                                                  sqr = sqr,
                                                                  hybrid_verification = hybrid_verification)          
    else :
        all_X, all_x, nb_pts = data.pattern_detection(left_folder = left_folder,
                                                      right_folder = right_folder,
                                                      name = name,
                                                      saving_folder = saving_folder,
                                                      ncx = ncx,
                                                      ncy = ncy,
                                                      sqr = sqr,
                                                      hybrid_verification = hybrid_verification)        

    # Creation of the reference matrix Xref and the real position Ucam for 
    # each camera
    x, Xc1, Xc2 = data.camera_np_coordinates(all_X, all_x, z_list)     

    # Plot the references plans
    solvel.refplans(x, z_list, plotting = plotting)

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
    Soloff_constants0, Soloff_constants = A_0
    return(Soloff_constants0, Soloff_constants, Magnification)

def Soloff_identification (Xc1_identified : np.ndarray,
                           Xc2_identified : np.ndarray,
                           Soloff_constants0 : np.ndarray, 
                           Soloff_constants : np.ndarray,
                           Soloff_pform : int,
                           method : str = 'curve_fit') -> np.ndarray :
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
           Python method used to solve it ('Least-squares' or 'curve-fit')

    Returns:
       x_solution : numpy.ndarray
           Identification in the 3D space of the detected points
    """
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
    x0 = solvel.least_square_method (Xc1_identified, Xc2_identified, Soloff_constants0)
    
    # Solve the polynomials constants ai with curve-fit method (Levenberg 
    # Marcquardt)
    xsolution, Xc, Xd = solvel.Levenberg_Marquardt_solving(Xc1_identified, 
                                                           Xc2_identified, 
                                                           Soloff_constants, 
                                                           x0, 
                                                           Soloff_pform, 
                                                           method = 'curve_fit')
    if modif_22_12_09 :
        xsolution = xsolution.reshape((3, nx, ny))
    return (xsolution)

def direct_calibration (__calibration_dict__ : dict,
                        z_list : np.ndarray,
                        direct_pform : int,
                        hybrid_verification : bool = False,
                        multifolder : bool = False,
                        plotting : bool = False) -> (np.ndarray, 
                                                     np.ndarray) :
    """Calculation of the magnification between reals and detected positions 
    and the calibration parameters A:--> x = A.M(X)
    
    Args:
       __calibration_dict__ : dict
           Calibration properties define in a dict. Including 'left_folder', 
           'right_folder', 'name', 'ncx', 'ncy', 'sqr'
       z_list : numpy.ndarray
           List of the different z position. (WARNING : Should be order the 
                                              same way in the target folder)
       direct_pform : int
           Polynomial degree
       hybrid_verification : bool, optional
           If True, verify each pattern detection and propose to pick 
           manually the bad detected corners. The image with all detected
           corners is show and you can decide to change any point using
           it ID (ID indicated on the image) as an input. If there is no
           bad detected corner, press ENTER to go to the next image.
       multifolder : bool, optional
           Used for specific image acquisition when all directions moved
       plotting = Bool
           Plot the calibration view or not
           
    Returns:
       direct_constants : numpy.ndarray
           Constants of direct polynomial
       Magnification : numpy.ndarray
           Magnification between reals and detected positions
    """
    z_list = np.array(z_list)
    
    if direct_pform == 1 :
        direct_constants = np.zeros((3, 5))
    elif direct_pform == 2 :
        direct_constants = np.zeros((3, 15))
    elif direct_pform == 3 :
        direct_constants = np.zeros((3, 35))
    elif direct_pform == 4 :
        direct_constants = np.zeros((3, 70))
    elif direct_pform == 5 :
        direct_constants = np.zeros((3, 121))
    else :
        raise ('Only define for polynomial degrees (1, 2, 3, 4 or 5')
    
    # Detect points from folders
    if multifolder :
        all_X, all_x, nb_pts = data.multifolder_pattern_detection(__calibration_dict__,
                                                                  hybrid_verification = hybrid_verification)          
    else :
        all_X, all_x, nb_pts = data.pattern_detection(__calibration_dict__,
                                                      hybrid_verification = hybrid_verification)      

    # Creation of the reference matrix Xref and the real position Ucam for 
    # each camera i
    x, Xc1, Xc2 = data.camera_np_coordinates(all_X, all_x, z_list)

    # Plot the references plans
    solvel.refplans(x, z_list, plotting = plotting)
        
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

    return(direct_constants, Magnification)    

def direct_calibration2 (z_list : np.ndarray,
                         direct_pform : int,
                         left_folder : str = 'left_calibration',
                         right_folder : str = 'right_calibration',
                         name : str = 'calibration',
                         saving_folder : str = 'results',
                         ncx : int = 16,
                         ncy : int = 12,
                         sqr : float = 0.3,
                         hybrid_verification : bool = False,
                         multifolder : bool = False,
                         plotting : bool = False) -> (np.ndarray, 
                                                      np.ndarray) :
    """Calculation of the magnification between reals and detected positions 
    and the calibration parameters A:--> x = A.M(X)
    
    Args:
       z_list : numpy.ndarray
           List of the different z position. (WARNING : Should be order the 
                                              same way in the target folder)
       direct_pform : int
           Polynomial degree
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
       sqr : float, optional
           Size of a square (in mm)
       hybrid_verification : bool, optional
           If True, verify each pattern detection and propose to pick 
           manually the bad detected corners. The image with all detected
           corners is show and you can decide to change any point using
           it ID (ID indicated on the image) as an input. If there is no
           bad detected corner, press ENTER to go to the next image.
       multifolder : bool, optional
           Used for specific image acquisition when all directions moved
       plotting = Bool
           Plot the calibration view or not
           
    Returns:
       direct_constants : numpy.ndarray
           Constants of direct polynomial
       Magnification : numpy.ndarray
           Magnification between reals and detected positions
    """
    z_list = np.array(z_list)
    
    if direct_pform == 1 :
        direct_constants = np.zeros((3, 5))
    elif direct_pform == 2 :
        direct_constants = np.zeros((3, 15))
    elif direct_pform == 3 :
        direct_constants = np.zeros((3, 35))
    elif direct_pform == 4 :
        direct_constants = np.zeros((3, 70))
    elif direct_pform == 5 :
        direct_constants = np.zeros((3, 121))
    else :
        raise ('Only define for polynomial degrees (1, 2, 3, 4 or 5')
    
    # Detect points from folders
    if multifolder :
        all_X, all_x, nb_pts = data.multifolder_pattern_detection(left_folder = left_folder,
                                                                  right_folder = right_folder,
                                                                  name = name,
                                                                  saving_folder = saving_folder,
                                                                  ncx = ncx,
                                                                  ncy = ncy,
                                                                  sqr = sqr,
                                                                  hybrid_verification = hybrid_verification)          
    else :
        all_X, all_x, nb_pts = data.pattern_detection(left_folder = left_folder,
                                                      right_folder = right_folder,
                                                      name = name,
                                                      saving_folder = saving_folder,
                                                      ncx = ncx,
                                                      ncy = ncy,
                                                      sqr = sqr,
                                                      hybrid_verification = hybrid_verification)       

    # Creation of the reference matrix Xref and the real position Ucam for 
    # each camera i
    x, Xc1, Xc2 = data.camera_np_coordinates(all_X, all_x, z_list)

    # Plot the references plans
    solvel.refplans(x, z_list, plotting = plotting)
        
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

    return(direct_constants, Magnification)    

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
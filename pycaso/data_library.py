#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from glob import glob
from copy import deepcopy
import DIC
import math
import matplotlib.pyplot as plt
import sys
import os
import time
import skimage.feature as sfe
import skimage.filters as sfi
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border

# Add the path of GCpu library if you want
sys.path.append('/home/caroneddy/These/GCpu_OpticalFlow-master/Src')

try : 
    from compute_flow import compute_flow
except ImportError:
    print('No module named compute_flow')

try : 
    import cupy 
    cpy = True
except ImportError:
    cpy = False
import numpy as np

import cv2

def calibrate(im : str ,
              ncx : int = 16,
              ncy : int = 12,
              sqr : float = 0.3):
    """ Detection of the corners
    
    Args:
        im : str
            Image path to detect
        ncx : int, optional
            The number of squares for the chessboard through x direction
        ncy : int, optional
            The number of squares for the chessboard through y direction
        sqr : float, optional
            Size of a square (in mm)
           
    Returns:
        corners_list : list (Dim = N * 3)
            List of the detected corners 
        pts : list
            Number of detected corners
    """
    mrk = sqr / 2
    if cv2.__version__=='4.7.0' :
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        parameters = cv2.aruco.DetectorParameters()
        parameters.adaptiveThreshWinSizeMax = 300
        board = cv2.aruco.CharucoBoard((ncx, ncy),
                                       sqr,
                                       mrk,
                                       dictionary)
    else :
        dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        parameters = cv2.aruco.DetectorParameters_create()
        parameters.adaptiveThreshWinSizeMax = 300
        board = cv2.aruco.CharucoBoard_create(ncx,
                                              ncy,
                                              sqr,
                                              mrk,
                                              dictionary)
    if len (im) < 20 :
        print("=> Calculation of the image ...", str(im))
    else :
        print("=> Calculation of the image ...", str(im[-20:]))
    img = cv2.imread(im, 0)        
    corners, ids, rip = cv2.aruco.detectMarkers(img, 
                                                dictionary, 
                                                parameters = parameters)
    
    idall = []
    nids = int(ncx*ncy/2)
    if len(corners) != 0 :
        if len(corners) < nids :
            for idd in list(range(0, nids)):
                if idd not in ids:
                    idall.append(int(idd))
        
        print("marks ", idall, " not detected")         
        if ids is not None and len(ids) > 0:
            pts, chcorners, chids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, img, board)
            print(len(corners), ' marks detected. ', pts, ' points detected')
            corners_list = []
            BU = []
            if pts != 0 :
                for i in range (0, len(chcorners)) :
                    BU.append(chcorners[i][0])
                    corners_list.append([BU[i][0],BU[i][1],chids[i][0]])
            else :
                ()
    else :
        corners_list = [False]
        pts = 0
        print("No marks detected")         
    return (corners_list, pts) 

def complete_missing_points (corners_list : np.ndarray, 
                             im : str, 
                             ncx : int = 16,
                             ncy : int = 12,
                             sqr : float = 0.3,
                             hybrid_verification : bool = False) -> list :  
    """ Detection of the corners with Hessian invariants filtering
    
    Args:
        corners_list : numpy.array
            Array of the detected points (automatically with ChAruco) 
        im : str
            Image path to detect
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

           
    Returns:
        corners_list_opt : list (Dim = N * 3)
            List of the detected corners (automatically with ChAruco and 
                                          Hessian invariants + manually)
    """ 
    corners_list = np.asarray(corners_list)
    x, y, ids = np.transpose(corners_list)
    img = cv2.imread(im, 0)

    # Filter the image with the Hessian matrix parameters to detect the 
    # corners (points)
    img_hess = plt.imread(im)
    if len (img_hess.shape) == 3 :
        img_hess = img_hess[:,:,0]
    HE0, HE1 = sfe.hessian_matrix_eigvals(sfe.hessian_matrix(img_hess, 9))
    HE = abs(HE0 * HE1)
    thresh = sfi.threshold_otsu(HE)
    bin_im = HE > thresh
    
    # Plot the already detected points
    if hybrid_verification :
        fig0, ax = plt.subplots()
        plt.imshow(img_hess, cmap='gray')
        plt.scatter(x,y, c='r')
        for name, txt in enumerate(ids):
            ax.annotate(txt, (x[name], y[name]))
    
    # Choose 2 points A and B already detected that could create 
    # the referential
    nx, ny = ncx-1, ncy-1
    n_corners = nx*ny
    corners_list_opt = np.zeros((n_corners, 3))
    pts_list = np.arange(n_corners)
    pts_list = np.reshape(pts_list, (ny,nx))
    ptA = corners_list[0]
    xA, yA, idA = ptA
    lineA, columnA = np.where(pts_list==idA)
    # Recreate the function numpy.delete for cupy
    def delete_arr (arr0, obj, axis) :
        nxarr, nyarr = arr0.shape
        obj = int(obj)
        if axis == 0 :
            arr1 = arr0[:obj]
            arr2 = arr0[obj+1:]
            arrt = np.append(arr1, arr2)
            arrt = arrt.reshape((nxarr-1,nyarr))
        elif axis == 1 :
            arr1 = np.transpose(arr0[:,:obj])
            arr2 = np.transpose(arr0[:,obj+1:])
            arrt = np.append(arr1, arr2)
            arrt = arrt.reshape((nyarr-1,nxarr))
            arrt = np.transpose(arrt)
        return(arrt)

    pts_list_cut = delete_arr(pts_list, lineA, 0)
    pts_list_cut = delete_arr(pts_list_cut, columnA, 1)
    pts_list_cut = np.ravel(pts_list_cut)
    ptB = []
    out_of_range_points = 0
    for pt in np.flip(pts_list_cut) :
        if np.any(corners_list[:,2] == pt) :
            lineB, columnB = np.where(pts_list == pt)
            line, column = np.where(corners_list == pt)
            ptB = corners_list[line]
            xB, yB, idB = ptB[0]
            break
        
    if np.any(ptB) :
        # Define the referencial coordinates of the pattern grid
        if cv2.__version__[:3] == '4.6' or cv2.__version__[:3] == '4.7' :
            CAB = columnB - columnA
            LAB = lineB - lineA
            xAB = xB - xA
            yAB = yB - yA
            dP = math.sqrt(xAB**2 + yAB**2)
            l = dP / math.sqrt(CAB**2 + LAB**2)
            
            Lx = (yAB-LAB*xAB/CAB)/(-CAB-LAB**2/CAB)
            Cy = -Lx
            Cx = (xAB -LAB*Lx)/CAB
            Ly = Cx
        
        elif cv2.__version__[:3] == '4.5' or cv2.__version__[:3] == '4.4'  :
            alpha = math.atan(-yAB/xAB)
            if xAB < 0 :
                alpha += math.pi
            alpha2 = math.atan(LAB/CAB)
            if CAB < 0 :
                alpha2 += math.pi
            alpha1 = alpha - alpha2
            Cx = l * math.cos(alpha1)
            xy = - l * math.sin(alpha1)
            Ly = - l * math.cos(alpha1)
            Lx = - l * math.sin(alpha1)
    
        # Define the origine point
        d0x = columnA * Cx + lineA * Lx
        d0y = columnA * Cy + lineA * Ly
        x0 = xA - d0x
        y0 = yA - d0y
    
        # Def win resize
        def win_spot (bin_im, l, d, xi, yi) :
            xm, ym = bin_im.shape
            if (xm < int(yi+d)) or (ym < int(xi+d)) or (0> int(yi-d)) or (0> int(xi-d)) :
                bary, barx = np.nan, np.nan
                max_area = np.nan
            else :
                # Try Hessian detection and pick the biggest binary 
                # area
                areas = []
                while areas == [] and d < l :
                    bin_im_win = bin_im[yi-d:yi+d, xi-d:xi+d]
                    # label_img=label(clear_border(bin_im_win))
                    label_img=label(bin_im_win)
                    regions = regionprops(label_img)
                    if len(regions) == 0 :
                        bary, barx = np.nan, np.nan
                        max_area = np.nan
                        print('Lose')
                        break
                    else :
                        for region in (regions):
                            areas.append(region.area)
                        if any (areas) :
                            max_area = max(areas)
                            max_i = areas.index(max_area)
                            region = regions[max_i]
                            bary, barx = region.centroid
                        else :
                            bary, barx = np.nan, np.nan
                            d += int(l//8)
            y_dot = bary + yi - d
            x_dot = barx + xi - d
            return(x_dot, y_dot, max_area)
        
        # Find the size of pattern area
        x_dot, y_dot, area_test = win_spot (bin_im, int(l), int(l*2/3), int(xA), int(yA))
        len_test = math.sqrt(area_test)
        
        # Find the holes
        for id_ in np.ravel(pts_list) :
            # Find the missing point
            line2, column2 = np.where(pts_list == id_)
            dix = column2 * Cx + line2 * Lx
            diy = column2 * Cy + line2 * Ly
            xi = int(x0 + dix)
            yi = int(y0 + diy)
            d = int(len_test)
            
            # Find the missing point, if on the screen
            x_dot, y_dot, __ = win_spot (bin_im, l, d, xi, yi)
            if not math.isnan(x_dot) :
                # Do it again around center 
                x_dot, y_dot, __ = win_spot (bin_im, l, d, int(x_dot), int(y_dot))
            
            if math.isnan(x_dot) :
                out_of_range_points += 1
            
            arr = np.array([float(x_dot), float(y_dot), float(id_)])
            if hybrid_verification :
                plt.annotate(id_, (x_dot, y_dot))
                plt.scatter(x_dot, y_dot, c='b', label = 'Hessian')

            corners_list_opt[id_] = arr

        while hybrid_verification :
            plt.pause(0.001)
            print('')
            print('Choose a bad detected corner if any. If None is, press Enter')
            txt = input()
            if txt =='' :
                print('End correction')
                plt.close()
                break
            else :
                if any (txt) in corners_list_opt[:,2] :
                    # If the Hessian detection is bad, manualy detection
                    # missing_points = [0, 0]
                    def onclick(event):
                        global missing_points
                        missing_points = [event.xdata, event.ydata]
                        plt.close()
                    id_ = int(txt)
                    line2, column2 = np.where(pts_list == id_)
                    dix = column2 * Cx + line2 * Lx
                    diy = column2 * Cy + line2 * Ly
                    xi = int(x0 + dix)
                    yi = int(y0 + diy)
                    fig, ax = plt.subplots()
                    plt.imshow(img[int(yi-d):int(yi+d), int(xi-d):int(xi+d)], cmap='gray')
                    fig.canvas.mpl_connect('button_press_event', onclick)
                    plt.title('Click on the missing corner')
                    plt.show()
                    plt.waitforbuttonpress()
                    plt.pause(0.001)
                    xi = xi+missing_points[0]-d
                    yi = yi+missing_points[1]-d
                    fig, ax = plt.subplots()
                    plt.imshow(img[int(yi-10):int(yi+10), int(xi-10):int(xi+10)], cmap='gray')
                    fig.canvas.mpl_connect('button_press_event', onclick)
                    plt.title('Click again')
                    plt.show()
                    plt.waitforbuttonpress()
                    xi = xi+missing_points[0]-10
                    yi = yi+missing_points[1]-10
                    arr = np.array([xi, yi, id_])
                    print('arr ', arr)
                    # print(corners_list)
                    corners_list_opt[id_] = arr
                    fig0
                    plt.scatter(xi,yi,c='g')
                else :
                    print('No corner with the id ', txt, ' chose another one')

                
    else :
        print('Impossible to detect manualy corners of image : ', im)
        corners_list_opt = np.array([False])
    print (out_of_range_points, ' points out of the image or to close to the border')
    print ('---')
    return (corners_list_opt)

def calibration_model(nx, 
                      ny, 
                      l) : 
    """ Creation of the model of the calibration pattern
    
    Args:
        nx : int
            Number of x squares
        ny : int
            Number of y squares
        l : int
            Size of a square
        
    Returns:
        Xref : list (Dim = N * 3)
            List of the real corners
            
    """
    Xref = []
    for i in range (0, ny-1) :
        for j in range (0, nx-1) :
            Xref.append([(nx-(j+1))*l, (i+1)*l, j+(ny-1)*i])
    return Xref

def cut_calibration_model (List_images : list , 
                           Xref : list , 
                           ncx : int = 16,
                           ncy : int = 12,
                           sqr : float = 0.3) -> (np.ndarray,
                                                  np.ndarray) :
    """ Group all of the images detected and filter the points not detected. 
        For each corners not detected on an image, delete it on all the others 
        images. 
        Delete also on the real positions of the corners.
    
    Args:
        List_images : list
            List of the detected corners
        Xref : list
            List of the real corners
        ncx : int, optional
            The number of squares for the chessboard through x direction
        ncy : int, optional
            The number of squares for the chessboard through y direction
        sqr : float, optional
            Size of a square (in mm)
        
    Returns:
        all_x : np.ndarray (Dim = Nimages * N * 3)
            List of the real corners
        all_X : np.ndarray (Dim = Nimages * N * 3)
            List of the detected corners
            
    """    
    Ucam_init = []
    # Xref = []
    holes = [] # List of the missing points
    M = len(List_images)
    
    # First, detect the holes = missing points
    nb_pts = np.zeros(M)
    print('M ', M)
    for i in range (0, M) :
        B, pts = calibrate(sorted(glob(List_images[i]))[0],
                           ncx = ncx,
                           ncy = ncy,
                           sqr = sqr)
        Ucam_init.append(B)
        nb_pts[i] = pts
        N = len(B)
        points = []
        for j in range (0, N) :
            points.append(B[j][2])
        Nall = len(Xref)
        holes_j = [j for j in range(Nall)]
        for j in range (0, N) :
            p = points[N-(j+1)]
            del(holes_j[p])
        holes = holes + holes_j 
    
    # Then arrange the holes and delect twins
    holes = list(dict.fromkeys(holes))
    holes = sorted(holes)
    T = len(holes)
    
    # Then delete those holes on all_X
    all_X = []
    for i in range (0, M) :     
        j = 0
        Ucam_remove = deepcopy(Ucam_init[i])
        for t in range (0, T) :
            N = len (Ucam_remove)
            button = True
            while j < N and button == True :                
                if Ucam_remove[j][2] == holes[t] :
                    del(Ucam_remove[j])
                    button = False
                elif int(Ucam_remove[j][2]) > int(holes[t]) :
                    button = False                
                else :
                    j += 1
        all_X.append(Ucam_remove)
        
    Pmax = len(Xref)
    print('----------')
    print(str(T) + ' points deleted in each images on a total of ' + str(Pmax) + ' points')
    print('----------')
          
    # Then delete those holes on all_x
    x = deepcopy(Xref)
    for t in range (0, T) :     
        p = holes[T-(t+1)]
        del(x[p])
    all_x = []
    for i in range (0, M) :
        all_x.append(x)
        

    if all_X[0] == [] :
        ()
    else :
        # Use it as array
        all_X = np.asarray(all_X)
        all_X = all_X[:, :, [0, 1]]
        all_x = np.asarray(all_x)
        all_x = all_x[:, :, [0, 1]]
    nb_pts = nb_pts.reshape((2, M//2))
    return (all_x, all_X, nb_pts)

def NAN_calibration_model (Images : list , 
                           Xref : list , 
                           ncx : int = 16,
                           ncy : int = 12,
                           sqr : float = 0.3,
                           hybrid_verification : bool = False) -> (np.ndarray,
                                                                   np.ndarray) :
    """ Group all of the images detected and filter the points not detected. 
        For each corners not detected on an image, replace the points with NAN. 
    
    Args:
        Images : list
            List of the detected corners
        Xref : list
            List of the real corners
        ncx : int, optional
            The number of squares for the chessboard through x direction
        ncy : int, optional
            The number of squares for the chessboard through y direction
        sqr : float, optional
            Size of a square (in mm)
        pattern : str
            Name of the pattern used ('macro' or 'micro')
        hybrid_verification : bool, optional
            If True, verify each pattern detection and propose to pick 
            manually the bad detected corners. The image with all detected
            corners is show and you can decide to change any point using
            it ID (ID indicated on the image) as an input. If there is no
            bad detected corner, press ENTER to go to the next image.
        
    Returns:
        all_x : np.ndarray (Dim = Nimages * N * 3)
            Array of the real corners
        all_X : np.ndarray (Dim = Nimages * N * 3)
            Array of the detected corners
            
    """    
    
    M = len(Images)
    
    # First, detect the holes = missing points
    Nall = len(Xref)
    nb_pts = np.zeros(M)
    all_X = np.zeros((M, Nall, 3))
    for i in range (0, M) :
        im = sorted(glob(Images[i]))[0]
        corners_list, pts = calibrate(im,
                                      ncx = ncx,
                                      ncy = ncy,
                                      sqr = sqr)
        nb_pts[i] = pts
        if any (corners_list) :
            corners_list = complete_missing_points(corners_list, 
                                                   im,
                                                   ncx = ncx,
                                                   ncy = ncy,
                                                   sqr = sqr,
                                                   hybrid_verification = hybrid_verification)
            if np.any(corners_list) :
                ()
            else :
                corners_list = np.empty((Nall, 3))
                corners_list[:] = np.nan
        else :
            corners_list = np.empty((Nall, 3))
            corners_list[:] = np.nan
        
        all_X[i] = corners_list

    all_x = []
    for i in range (0, M) :
        all_x.append(Xref)        
    # Use it as array
    all_x = np.asarray(all_x)
    all_x = all_x[:, :, [0, 1]]
    all_X = all_X[:, :, [0, 1]]
    nb_pts = np.reshape(nb_pts, (2, M//2))
    return (all_x, all_X, nb_pts)

def pattern_detection (left_folder : str = 'left_calibration',
                       right_folder : str = 'right_calibration',
                       name : str = 'calibration',
                       saving_folder : str = 'results',
                       ncx : int = 16,
                       ncy : int = 12,
                       sqr : float = 0.3,
                       hybrid_verification : bool = False,
                       save : bool = True) -> (np.ndarray,
                                               np.ndarray) :
    """Detect the corners of Charucco's pattern.
    
    Args:
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
       save : bool, optional
           Save the datas in the saving_folder
           
    Returns:
       all_X : numpy.ndarray
           The corners of the pattern detect by the camera ranged in an array 
           arrange with all left pictures followed by all right pictures. 
           Expl : [left_picture_1, left_picture_2, right_picture_1, 
                   right_picture_2]
       all_x : numpy.ndarray
           The theorical corners of the pattern
    """
    # Taking the main parameters from bibliotheque_data_eddy.
    Images_left = sorted(glob(str(left_folder) + '/*'))
    Images_right = sorted(glob(str(right_folder) + '/*'))
    Images = Images_left
    for i in range (len(Images_right)) :
        Images.append(Images_right[i])
    
    Save_all_X = str(saving_folder) + "/all_X_" + name + ".npy"
    Save_all_x = str(saving_folder) + "/all_x_" + name + ".npy"
    Save_nb_pts = str(saving_folder) + "/nb_pts_" + name + ".npy"
    
    # Corners detection
    if os.path.exists(Save_all_X) and os.path.exists(Save_all_x) and os.path.exists(Save_nb_pts) :
        # Taking pre-calculated datas from the saving_folder
        print('    - Taking datas from ', saving_folder)        
        all_X = np.load(Save_all_X)
        all_x = np.load(Save_all_x)
        nb_pts = np.load(Save_nb_pts)
    
    else : # Corners detection
        print('    - Detection of the pattern in progress ...')
        # Creation of the theoretical pattern + detection of camera's pattern
        Xref = calibration_model(ncx, ncy, sqr)
        all_x, all_X, nb_pts = NAN_calibration_model(Images, 
                                                     Xref, 
                                                     ncx = ncx,
                                                     ncy = ncy,
                                                     sqr = sqr,
                                                     hybrid_verification = hybrid_verification)

        if not np.any(all_X[0]):
            print('Not any point detected in all images/cameras')
        else :
            if save :
                np.save(Save_all_X, all_X)
                np.save(Save_all_x, all_x)
                np.save(Save_nb_pts, nb_pts)
                print('    - Saving datas in ', saving_folder)

        
    return(all_X, all_x, nb_pts)

def multifolder_pattern_detection (left_folder : str = 'left_calibration',
                                   right_folder : str = 'right_calibration',
                                   name : str = 'calibration',
                                   saving_folder : str = 'results',
                                   ncx : int = 16,
                                   ncy : int = 12,
                                   sqr : float = 0.3,
                                   hybrid_verification : bool = False) -> (np.ndarray,
                                                                           np.ndarray) :
    """Detect the corners of Charucco's pattern in multiple folders.
    
    Args:
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
           
    Returns:
       all_X : numpy.ndarray
           The corners of the pattern detect by the camera ranged in an array 
           arrange with all left pictures followed by all right pictures. 
           Expl : [left_picture_1, left_picture_2, right_picture_1, 
                   right_picture_2]
       all_x : numpy.ndarray
           The theorical corners of the pattern
    """
    # Taking the main parameters from bibliotheque_data_eddy.
    nxy = len(sorted(glob(str(left_folder) + '/*')))
    nz = len(sorted(glob(str(sorted(glob(str(sorted(glob(str(left_folder) + '/*'))[0]) + '/*'))[0]) + '/*')))
    npts = (ncx-1)*(ncy-1)
    multall_X = np.empty((nxy, nxy, 2*nz, npts, 2))
    multall_x = np.empty((nxy, nxy, 2*nz, npts, 2))
    multnb_pts = np.empty((nxy, nxy, 2, nz))
    Save_all_X = str(saving_folder) + "/all_X_" + name + ".npy"
    Save_all_x = str(saving_folder) + "/all_x_" + name + ".npy"
    Save_nb_pts = str(saving_folder) + "/nb_pts_" + name + ".npy"
    # Corners detection
    if os.path.exists(Save_all_X) and os.path.exists(Save_all_x) and os.path.exists(Save_nb_pts) :
        # Taking pre-calculated datas from the saving_folder
        print('    - Taking datas from ', saving_folder)        
        all_X = np.load(Save_all_X)
        all_x = np.load(Save_all_x)
        nb_pts = np.load(Save_nb_pts)
    
    else : # Corners detection
        for i, imx in enumerate(sorted(glob(str(left_folder) + '/*'))) :
            dx = imx[len(left_folder)+2:]
            for j, imy in enumerate(sorted(glob(str(imx) + '/*'))) :
                dy = imy[len(imx)+2:]
                all_X, all_x, nb_pts = pattern_detection (left_folder = left_folder + imy[len(left_folder):],
                                                          right_folder = right_folder + imy[len(left_folder):],
                                                          name = name,
                                                          saving_folder = saving_folder,
                                                          ncx = ncx,
                                                          ncy = ncy,
                                                          sqr = sqr,
                                                          hybrid_verification = hybrid_verification,
                                                          save = False)
                all_x[:,:,0] += float(dx)
                all_x[:,:,1] += float(dy)
                
                multall_X [i,j]= all_X
                multall_x [i,j]= all_x
                multnb_pts [i,j]= nb_pts
                print('x = ', dx)
                print('y = ', dy)
        
        # Re_organise arrays
        all_X = np.empty((2*nz, nxy*nxy*npts, 2))
        all_x = np.empty((2*nz, nxy*nxy*npts, 2))
        nb_pts = np.empty((2, nz))
        for i in range (nz) :
            all_X[i] = multall_X[:,:,i,:,:].reshape((nxy*nxy*npts, 2))
            all_x[i] = multall_x[:,:,i,:,:].reshape((nxy*nxy*npts, 2))
            
            all_X[i+nz] = multall_X[:,:,i+nz,:,:].reshape((nxy*nxy*npts, 2))
            all_x[i+nz] = multall_x[:,:,i+nz,:,:].reshape((nxy*nxy*npts, 2))
            nb_pts[0, i] = np.min(multnb_pts[:,:,0,i])
            nb_pts[1, i] = np.min(multnb_pts[:,:,1,i])
        
        # Save datas
        if not np.any(all_X[0]):
            print('Not any point detected in all images/cameras')
        else :
            np.save(Save_all_X, all_X)
            np.save(Save_all_x, all_x)
            np.save(Save_nb_pts, nb_pts)
            print('    - Saving datas in ', saving_folder)
        
    return(all_X, all_x, nb_pts)

def hybrid_mask_creation (image : np.ndarray,
                          ROI : bool = False,
                          kernel : int = 5,
                          gate : int = 5) -> list :
    """Create a mask with the function Otsu from skimage
    
    Args:
       image : numpy.ndarray
           Difference between direct and Soloff methods
       ROI : str, optional
           Region Of Interest
       kernel : int, optional
           Size of smoothing filter
       gate : int, optional
           Output value (in µm) where the mask is True
           
    Returns:
       mask_median : list
           Mask used to replace on direct method + the median of the difference
           between Soloff and direct solutions
    """      
    median = np.median(image)     
    kernel = np.ones((kernel,kernel),np.float32)/kernel**2
    image_smooth = cv2.filter2D(image,-1,kernel)
    if ROI :
        x1, x2 = ROI[0]
        y1, y2 = ROI[1]
        image_crop = image_smooth * 0
        image_crop[x1:x2,y1:y2] = image_smooth[x1:x2,y1:y2]
    else :
        image_crop = image_smooth

    inside_mask = np.ma.masked_inside(image_smooth*1000, -gate, gate)
    mask_median = [inside_mask.mask, median]
    return (mask_median)

def camera_np_coordinates (all_X : np.ndarray, 
                           all_x : np.ndarray, 
                           z_list : np.ndarray) -> (np.ndarray,
                                                    np.ndarray,
                                                    np.ndarray) :
    """Organising the coordinates of the calibration
    
    Args:
       all_X : numpy.ndarray
           The corners of the pattern detect by the camera
       all_x : numpy.ndarray
           The theorical corners of the pattern
       z_list : numpy.ndarray
           List of the different z position. (Ordered the same way in the 
           target folder)
    Returns:
       x : numpy.ndarray
           Organised real positions in 3D space
       Xc1 : numpy.ndarray
           Organised detected positions of camera 1
       Xc2 : numpy.ndarray
           Organised detected positions of camera 2
    """
    for i in [1, 2] :
        print('')
        mid = all_X.shape[0]//2    
        all_Xi = all_X[(i-1)*mid:i*mid,:,:]
        all_xi = all_x[i*(mid-1):i*mid,:,:]
        sU = all_Xi.shape
        Xref = all_xi[0]
        all_xi = np.empty ((sU[0], sU[1], sU[2]+1))
        x = np.empty ((sU[0] * sU[1], sU[2]+1))
        X = np.empty ((sU[0] * sU[1], sU[2]))
        for j in range (sU[0]) :
            all_xi[j][:,0] = Xref[:,0]
            all_xi[j][:,1] = Xref[:,1]
            all_xi[j][:,2] = z_list[j]

            x[j*sU[1] : (j+1)*sU[1], :]  = all_xi[j]
            X[j*sU[1] : (j+1)*sU[1], :]  = all_Xi[j]

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
            Xc1 = X
        if i == 2 :
            Xc2 = X
    # If there is some NAN value, then delete all 2D and 3D corresponding 
    # points
    if np.isnan(Xc1).any() or np.isnan(Xc2).any() :
        mask1 = np.ma.masked_invalid(Xc1)
        mask2 = np.ma.masked_invalid(Xc2)
        mask = mask1.mask + mask2.mask
        Xc1 = Xc1[np.logical_not(mask)]
        Xc1 = np.reshape(Xc1, (2, len(Xc1)//2))
        Xc2 = Xc2[np.logical_not(mask)]
        Xc2 = np.reshape(Xc2, (2, len(Xc2)//2))
        mask = mask[0]
        x1 = x1[np.logical_not(mask)]
        x2 = x2[np.logical_not(mask)]
        x3 = x3[np.logical_not(mask)]
        x = np.asarray([x1,x2,x3])
    else :
        mask = np.array([False])
        
    return (x, Xc1, Xc2)

def DIC_disflow (DIC_dict : dict,
                 flip : bool = False,
                 image_ids : list = [False]) -> (np.ndarray,
                                                 np.ndarray) :
    """Use the DIC to locate all the points from the reference picture
    (first left one) in the deformed ones (other left and right pictures).
    
    Args:
       DIC_dict : dict
           DIC dictionnary including the picture folders, the saving name and 
           the window (in px) to study.
       flip : bool, optional
           If True, all the pictures are flipped before the DIC (useful when 
           you're using a mirror)
       image_ids : list, optional
           Define the list of images you want to compare in the left and right folders
           
    Returns:
       Xleft_id : numpy.ndarray
           All the points of the left pictures (1 point per pixel) in an array 
           arrange with their positions. 
       Xright_id : numpy.ndarray
           All the left pixels (points) localised on the right pictures.
    """
    left_folder = DIC_dict['left_folder']
    right_folder = DIC_dict['right_folder']
    window = DIC_dict['window']
    vr_kwargs = DIC_dict['dic_kwargs'] if 'dic_kwargs' in DIC_dict else ()
    
    Images_left = sorted(glob(str(left_folder) + '/*'))
    Images_right = sorted(glob(str(right_folder) + '/*'))
    if any (image_ids) :
        Images_left_cut = []
        Images_right_cut = []
        for i in image_ids :
            Images_left_cut.append(Images_left[i])
            Images_right_cut.append(Images_right[i])
        Images_left = Images_left_cut
        Images_right = Images_right_cut
    Images = Images_left
    N = len(Images)
    for i in range (N) :
        Images.append(Images_right[i]) 
    [lx1, lx2], [ly1, ly2] = window
    N = len(Images)

    print('    - DIC in progress ...')
    name = DIC_dict['name']
    if any (image_ids) :
        for i in image_ids :
            name = name + str(i)
    Save_all_U = str(DIC_dict['saving_folder']) +"/compute_flow_U_" + name + ".npy"
    Save_all_V = str(DIC_dict['saving_folder']) +"/compute_flow_V_" + name + ".npy"
    if os.path.exists(Save_all_U) and os.path.exists(Save_all_V):
        print('Loading data from\n\t%s\n\t%s' % (Save_all_U, Save_all_V))
        all_U = np.load(Save_all_U)
        all_V = np.load(Save_all_V)
    else:
        im0 = cv2.imread(Images[0], 0)
        all_U = np.zeros((N, im0.shape[0], im0.shape[1]))
        all_V = np.zeros((N, im0.shape[0], im0.shape[1]))
        for i in range(1, N):
            print('\nComputing flow between\n\t%s\n\t%s' % (Images[0], Images[i]))
            Im1 = cv2.imread(Images[0],0) 
            Im2 = cv2.imread(Images[i],0) 
            if flip :
                Im1 = cv2.flip(Im1, 1)
                Im2 = cv2.flip(Im2, 1)
            all_U[i], all_V[i] = DIC.displacement_field(Im1, 
                                                        Im2,
                                                        vr_kwargs=vr_kwargs)

        np.save(Save_all_U, all_U)
        np.save(Save_all_V, all_V)

    Xleft_id = []
    Xright_id = []        
    for i in range (N) :
        U, V = all_U[i], all_V[i]
        nX1, nX2 = U.shape
        linsp = np.arange(nX1)+1
        linsp = np.reshape (linsp, (1,nX1))
        X1matrix = np.matmul(np.ones((nX1, 1)), linsp)
        X2matrix = np.matmul(np.transpose(linsp), np.ones((1, nX1)))
        X1matrix_w = X1matrix[ly1:ly2, lx1:lx2]
        X2matrix_w = X2matrix[ly1:ly2, lx1:lx2]

        # Left camera --> position = each px
        X_c1 = np.transpose(np.array([np.ravel(X1matrix_w), 
                                      np.ravel(X2matrix_w)]))
        UV = np.transpose(np.array([np.ravel(U[ly1:ly2, lx1:lx2]), 
                                    np.ravel(V[ly1:ly2, lx1:lx2])]))

        # Right camera --> position = each px + displacement
        X_c2 = X_c1 + UV
        if i < N//2 :
            Xleft_id.append(X_c2)
        else : 
            Xright_id.append(X_c2)
    
    Xleft_id = np.array(Xleft_id)
    Xright_id = np.array(Xright_id)
    nim, npts, naxis = Xleft_id.shape
    Xleft_id = Xleft_id.reshape((nim, lx2-lx1, ly2-ly1, naxis))
    Xright_id = Xright_id.reshape((nim, lx2-lx1, ly2-ly1, naxis))
    return(Xleft_id, Xright_id)

def DIC_compute_flow (DIC_dict : dict,
                      flip : bool = False,
                      image_ids : list = [False]) -> (np.ndarray,
                                                      np.ndarray) :
    """Use the DIC to locate all the points from the reference picture
    (first left one) in the other ones (other left and right pictures).
    
    Args:
       DIC_dict : dict
           DIC dictionnary including the picture folders, the saving name and 
           the window (in px) to study.
       flip : bool, optional
           If True, all the pictures are flipped before the DIC (useful when 
           you're using a mirror)
       image_ids : list, optional
           Define the list of images you want to compare in the left and right folders
           
    Returns:
       Xleft_id : numpy.ndarray
           All the points of the left pictures (1 point per pixel) in an array 
           arrange with their positions. 
       Xright_id : numpy.ndarray
           All the left pixels (points) localised on the right pictures.
   """
    try : 
        from compute_flow import compute_flow
    except ImportError:
        raise('No module named compute_flow')

    left_folder = DIC_dict['left_folder']
    right_folder = DIC_dict['right_folder']
    ROI = DIC_dict['window']
    opt_flow = {"pyram_levels": 3, 
                "factor": 1/0.5, 
                "ordre_inter": 3, 
                "size_median_filter": 3, 
                "max_linear_iter": 1, 
                "max_iter": 10, 
                "lmbda": 2.*10**4, 
                "lambda2": 0.001, 
                "lambda3": 1., 
                "Mask": None,
                "LO_filter": 0}
    
    print('    - DIC in progress ...')
    if 'dic_kwargs' in DIC_dict :
        optical_flow_parameters = DIC_dict['dic_kwargs']  
    else :
        optical_flow_parameters = opt_flow

    Images_left = sorted(glob(str(left_folder) + '/*'))
    Images_right = sorted(glob(str(right_folder) + '/*'))
    if any (image_ids) :
        Images_left_cut = []
        Images_right_cut = []
        for i in image_ids :
            Images_left_cut.append(Images_left[i])
            Images_right_cut.append(Images_right[i])
        Images_left = Images_left_cut
        Images_right = Images_right_cut
    Images = Images_left
    N = len(Images)
    for i in range (N) :
        Images.append(Images_right[i]) 
    [lx1, lx2], [ly1, ly2] = ROI
    N = len(Images)

    name = DIC_dict['name']
    if any (image_ids) :
        for i in image_ids :
            name = name + str(i)
    Save_all_U = str(DIC_dict['saving_folder']) +"/compute_flow_U_" + name + ".npy"
    Save_all_V = str(DIC_dict['saving_folder']) +"/compute_flow_V_" + name + ".npy"
    if os.path.exists(Save_all_U) and os.path.exists(Save_all_V):
        print('Loading data from\n\t%s\n\t%s' % (Save_all_U, Save_all_V))
        all_U = np.load(Save_all_U)
        all_V = np.load(Save_all_V)
    else:
        im0_left = cv2.imread(Images[0], 0)
        im0_right = cv2.imread(Images[int(N/2)], 0)
        if flip :
            im0_left = cv2.flip(im0_left, 1)
            im0_right = cv2.flip(im0_right, 1)
        nx, ny = im0_left.shape
        all_U = np.zeros((N, nx, ny), dtype=np.float32)
        all_V = np.zeros((N, nx, ny), dtype=np.float32)
        x, y = np.meshgrid(np.arange(ny), np.arange(nx))
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        # Left0/left camera correlations + left0 / right0 correlation
        for i, image in enumerate(Images[1:(int(N/2)+1)]):
            im = cv2.imread(image, 0)
            if flip :
                im = cv2.flip(im, 1)
            print('\nComputing flow between\n\t%s\n\t%s' % (Images[0], image))
            t1 = time.time()
            U, V = compute_flow(im0_left, 
                                im, 
                                optical_flow_parameters["pyram_levels"], 
                                optical_flow_parameters["factor"], 
                                optical_flow_parameters["ordre_inter"],
                                optical_flow_parameters["lmbda"], 
                                optical_flow_parameters["size_median_filter"],
                                optical_flow_parameters["max_linear_iter"], 
                                optical_flow_parameters["max_iter"], 
                                optical_flow_parameters["lambda2"], 
                                optical_flow_parameters["lambda3"], 
                                optical_flow_parameters["Mask"], 
                                optical_flow_parameters["LO_filter"])
            try :
                import cupy
                U = cupy.asnumpy(U)
                V = cupy.asnumpy(V)
            except ImportError:
                ()  
            all_U[i+1], all_V[i+1] = U, V
            t2 = time.time()
            print('Elapsed time:', (t2-t1), '(s)  --> ', (t2-t1)/60, '(min)')
        # Left0/right camera composed correlations 
        for i, image in enumerate(Images[(int(N/2)+1):]):
            im = cv2.imread(image, 0)
            if flip :
                im = cv2.flip(im, 1)
            print('\nComputing flow between\n\t%s\n\t%s' % (Images[int(N/2)], image))
            t1 = time.time()
            # Right0/right camera correlation    
            Ur, Vr = compute_flow(im0_right,
                                  im,
                                  optical_flow_parameters["pyram_levels"],
                                  optical_flow_parameters["factor"],
                                  optical_flow_parameters["ordre_inter"],
                                  optical_flow_parameters["lmbda"],
                                  optical_flow_parameters["size_median_filter"],
                                  optical_flow_parameters["max_linear_iter"],
                                  optical_flow_parameters["max_iter"],
                                  optical_flow_parameters["lambda2"],
                                  optical_flow_parameters["lambda3"],
                                  optical_flow_parameters["Mask"],
                                  optical_flow_parameters["LO_filter"])
            try :
                import cupy
                Ur = cupy.asnumpy(Ur)
                Vr = cupy.asnumpy(Vr)
            except ImportError:
                ()
            Ur = Ur.astype(np.float32)
            Vr = Vr.astype(np.float32)
            t2 = time.time()
            print('Elapsed time:', (t2-t1), '(s)  --> ', (t2-t1)/60, '(min)')
            # Composition of transformations
            all_U[int(N/2)+i+1] = all_U[int(N/2)] + cv2.remap(Ur, x+all_U[int(N/2)], y+all_V[int(N/2)], cv2.INTER_LINEAR)
            all_V[int(N/2)+i+1] = all_V[int(N/2)] + cv2.remap(Vr, x+all_U[int(N/2)], y+all_V[int(N/2)], cv2.INTER_LINEAR)
        print('Saving data to\n\t%s\n\t%s' % (Save_all_U, Save_all_V))
        np.save(Save_all_U, all_U)
        np.save(Save_all_V, all_V)
    
    Xleft_id = []
    Xright_id = []
    for i in range (N) :
        U, V = all_U[i], all_V[i]
        nX1, nX2 = U.shape
        linsp1 = np.arange(nX1)+1
        linsp2 = np.arange(nX2)+1
        linsp1 = np.reshape (linsp1, (1,nX1))
        linsp2 = np.reshape (linsp2, (1,nX2))
        X1matrix = np.matmul(np.ones((nX1, 1)), linsp2)
        X2matrix = np.matmul(np.transpose(linsp1), np.ones((1, nX2)))
        X1matrix_w = X1matrix[lx1:lx2, ly1:ly2]
        X2matrix_w = X2matrix[lx1:lx2, ly1:ly2]

        # Left camera --> position = each px
        X_c1 = np.transpose(np.array([np.ravel(X1matrix_w), 
                                      np.ravel(X2matrix_w)]))
        UV = np.transpose(np.array([np.ravel(U[lx1:lx2, ly1:ly2]), 
                                    np.ravel(V[lx1:lx2, ly1:ly2])]))

        # Right camera --> position = each px + displacement
        X_c2 = X_c1 + UV
        if i < N//2 :
            Xleft_id.append(X_c2)
        else : 
            Xright_id.append(X_c2)
    
    Xleft_id = np.array(Xleft_id)
    Xright_id = np.array(Xright_id)
    nim, npts, naxis = Xleft_id.shape
    Xleft_id = Xleft_id.reshape((nim, lx2-lx1, ly2-ly1, naxis))
    Xright_id = Xright_id.reshape((nim, lx2-lx1, ly2-ly1, naxis))
    return (Xleft_id, Xright_id)

def DIC_get_positions (DIC_dict : dict,
                       flip : bool= False,
                       image_ids : list = [False],
                       method : str = 'compute_flow') -> (np.ndarray,
                                                          np.ndarray) :
    """Use the DIC to locate all the points from the reference picture
    (first left one) in the other ones (other left and right pictures).
    
    Args:
       DIC_dict : dict
           DIC dictionnary including the picture folders, the saving name and 
           the window (in px) to study.
       flip : bool, optional
           If True, all the pictures are flipped before the DIC (useful when 
           you're using a mirror)
       image_ids : list, optional
           Define the list of images you want to compare in the left and right folders
       method : str
           DIC method between compute_flow and disflow
           
    Returns:
       Xleft_id : numpy.ndarray
           All the points of the left pictures (1 point per pixel) in an array 
           arrange with their positions. 
       Xright_id : numpy.ndarray
           All the left pixels (points) localised on the right pictures.
    """
    if method == 'compute_flow' :
        try : 
            from compute_flow import compute_flow
        except ImportError:
            print('No module named conpute_flow, disflow from OpenCV will be used')
            method = 'disflow'
    if method == 'disflow' :
        return (DIC_disflow(DIC_dict,
                            flip = flip,
                            image_ids = image_ids))
    if method == 'compute_flow' :
        return (DIC_compute_flow(DIC_dict,
                                 flip = flip,
                                 image_ids = image_ids))
    else :
        print('No method known as ' + method + ', please chose "diflow" or "compute_flow"')
        raise

def DIC_get_positions2 (left_folder : str = 'left_identification',
                       right_folder : str = 'right_identification',
                       name : str = 'identification',
                       saving_folder : str = 'results',
                       window : list = [False],
                       flip : bool = False,
                       image_ids : bool = False,
                       method : str = 'compute_flow') ->(np.ndarray,
                                                         np.ndarray) :
    """Use the DIC to locate all the points from the reference picture
    (first left one) in the other ones (other left and right pictures).
    
    Args:
       left_folder : str, optional
           Left calibration images folder
       right_folder : str, optional
           Right calibration images folder
       name : str, optional
           identification
       saving_folder : str, optional
           results
       window : str, optional
           Window of the picture to process (in px)
       flip : bool, optional
           If True, all the pictures are flipped before the DIC (useful when 
           you're using a mirror)
       image_ids : list, optional
           Define the list of images you want to compare in the left and right folders
       method : str
           DIC method between compute_flow and disflow
           
    Returns:
       Xleft_id : numpy.ndarray
           All the points of the left pictures (1 point per pixel) in an array 
           arrange with their positions. 
       Xright_id : numpy.ndarray
           All the left pixels (points) localised on the right pictures.
    """
    DIC_dict = {
    'left_folder' : left_folder,
    'right_folder' : right_folder,
    'name' : name,
    'saving_folder' : saving_folder,
    'window' : window
    }
    if method == 'compute_flow' :
        try : 
            from compute_flow import compute_flow
        except ImportError:
            print('No module named conpute_flow, disflow from OpenCV will be used')
            method = 'disflow'
    if method == 'disflow' :
        return (DIC_disflow(DIC_dict,
                            flip = flip,
                            image_ids = image_ids))
    if method == 'compute_flow' :
        return (DIC_compute_flow(DIC_dict,
                                 flip = flip,
                                 image_ids = image_ids))
    else :
        print('No method known as ' + method + ', please chose "diflow" or "compute_flow"')
        raise

def DIC_fields (DIC_dict : dict,
                flip : bool = False) -> (np.ndarray,
                                         np.ndarray,
                                         np.ndarray,
                                         np.ndarray) :
    """Use the DIC to calcul all the left and right displacements fields.
    
    Args:
       DIC_dict : dict
           DIC dictionnary including the picture folders, the saving name and 
           the window (in px) to study.
       flip : bool, optional
           If True, all the pictures are flipped before the DIC (useful when 
           you're using a mirror)
           
    Returns:
       U_left : numpy.ndarray
           All the left displacements fields in x direction.
       V_left : numpy.ndarray 
           All the left displacements fields in y direction.
       U_right : numpy.ndarray 
           All the right displacements fields in x direction.
       V_right : numpy.ndarray
           All the right displacements fields in y direction.
    """
    saving_folder = DIC_dict['saving_folder']
    left_folder = DIC_dict['left_folder']
    right_folder = DIC_dict['right_folder']
    name = DIC_dict['name']
    Save_UV = str(saving_folder) +"/all_UV_" + name + ".npy"
    vr_kwargs = DIC_dict['dic_kwargs'] if 'dic_kwargs' in DIC_dict else ()
    
    # Taking pre-calculated datas from the saving_folder
    if os.path.exists(Save_UV) :
        print('    - Taking datas from ', saving_folder)        
        all_UV = np.load(Save_UV)
        U_left, V_left, U_right, V_right = all_UV
    else :
        Images_left = sorted(glob(str(left_folder) + '/*'))
        Images_right = sorted(glob(str(right_folder) + '/*'))
        Images = Images_left
        N = len(Images)
        for i in range (N) :
            Images.append(Images_right[i])    

        # Corners detection
        print('    - DIC in progress ...')
        # DIC detection of the points from each camera
        for i in range (N) :
            Iml1, Iml2 = Images[0], Images[i]
            Imr1, Imr2 = Images[N], Images[i+N]
            Iml1 = cv2.imread(Images[0],0) 
            Iml2 = cv2.imread(Images[i],0) 
            Imr1 = cv2.imread(Images[N],0) 
            Imr2 = cv2.imread(Images[i+N],0) 
            if flip :
                Iml1 = cv2.flip(Iml1, 1)
                Iml2 = cv2.flip(Iml2, 1)
                Imr1 = cv2.flip(Imr1, 1)
                Imr2 = cv2.flip(Imr2, 1)
            
            Ul, Vl = DIC.displacement_field(Iml1, 
                                            Iml2,
                                            vr_kwargs=vr_kwargs)
            Ur, Vr = DIC.displacement_field(Imr1, 
                                            Imr2,
                                            vr_kwargs=vr_kwargs)
            if i == 0 :
                U_left = np.zeros((N, Ul.shape[0], Ul.shape[1]))
                V_left = np.zeros((N, Ul.shape[0], Ul.shape[1]))
                U_right = np.zeros((N, Ul.shape[0], Ul.shape[1]))
                V_right = np.zeros((N, Ul.shape[0], Ul.shape[1]))
            U_left[i] = Ul
            V_left[i] = Vl
            U_right[i] = Ur
            V_right[i] = Vr
        all_UV = np.array([U_left, V_left, U_right, V_right])       
        np.save(Save_UV, all_UV)
        print('    - Saving datas in ', saving_folder)

    return(U_left, V_left, U_right, V_right)

def Strain_field (UVW : np.ndarray) -> (np.ndarray,
                                        np.ndarray,
                                        np.ndarray,
                                        np.ndarray,
                                        np.ndarray,
                                        np.ndarray):
    """Calcul all the strains field from displacements field
    
    Args:
       UVW : numpy.ndarray
           Displacements field
           
    Returns:
       Exy : numpy.ndarray
           strains field in %
       Exx : numpy.ndarray
           strains field in %
       Eyy : numpy.ndarray
           strains field in %
       Eyx : numpy.ndarray
           strains field in %
       Ezy : numpy.ndarray
           strains field in %
       Ezx : numpy.ndarray
           strains field in %
    """
    axis, nx, ny = UVW.shape
    Exyz = np.zeros((6, nx, ny))
    U, V, W = UVW
    Exyz[0], Exyz[1] = np.gradient(U)
    Exyz[2], Exyz[3] = np.gradient(V)
    Exyz[4], Exyz[5] = np.gradient(W)

    Exy, Exx, Eyy, Eyx, Ezy, Ezx = Exyz*100
            
    return(Exy, Exx, Eyy, Eyx, Ezy, Ezx)

def Strain_fields (UVW : np.ndarray) -> (np.ndarray,
                                         np.ndarray,
                                         np.ndarray,
                                         np.ndarray,
                                         np.ndarray,
                                         np.ndarray) :
    """Calcul all the strains fields from displacements fields
    
    Args:
       UVW : numpy.ndarray
           Displacements fields
           
    Returns:
       Exy : numpy.ndarray
           strains fields in %
       Exx : numpy.ndarray
           strains fields in %
       Eyy : numpy.ndarray
           strains fields in %
       Eyx : numpy.ndarray
           strains fields in %
       Ezy : numpy.ndarray
           strains fields in %
       Ezx : numpy.ndarray
           strains fields in %
    """
    Np_img, axis, nx, ny = UVW.shape
    Exy, Exx, Eyy, Eyx, Ezy, Ezx = [], [], [], [], [], []
    for i in range(Np_img) :
        Exyi, Exxi, Eyyi, Eyxi, Ezyi, Ezxi = Strain_field (UVW[i])
        Exy.append(Exyi)
        Exx.append(Exxi)
        Eyy.append(Eyyi)
        Eyx.append(Eyxi)
        Ezy.append(Ezyi)
        Ezx.append(Ezxi)
        
    return(Exy, Exx, Eyy, Eyx, Ezy, Ezx)
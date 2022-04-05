#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from glob import glob
from copy import deepcopy
import DIC
import math
import matplotlib.pyplot as plt
import sys
import pathlib
import os
from skimage.feature import structure_tensor, hessian_matrix
from skimage.filters import difference_of_gaussians, threshold_otsu
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border

try : 
    import cupy as np
except ImportError:
    import numpy as np
import cv2
import cv2.aruco as aruco


class Calibrate(dict):
    """Identification class of the corners of a chessboard by Charuco's method"""
    def __init__(self, _dict_):
        self._dict_ = _dict_
        # ncx, ncy, sqr, mrk = pattern_cst(pattern)
        self.ncx = _dict_['ncx']
        self.ncy = _dict_['ncy']
        self.sqr = _dict_['sqr']
        self.mrk = self.sqr / 2
        self.dictionary = aruco.Dictionary_get(aruco.DICT_6X6_250)
        self.parameters = aruco.DetectorParameters_create()
        self.parameters.adaptiveThreshWinSizeMax = 300
        self.board = aruco.CharucoBoard_create(
            self.ncx,
            self.ncy,
            self.sqr,
            self.mrk,
            self.dictionary)
    
    def calibrate(self, im):
        """ Detection of the corners
        
        Args:
            im : str
                Image path to detect
            
        Returns:
            corners_list : list (Dim = N * 3)
                List of the detected corners 
        """
        print("=> Calculation of the image {0}".format(im))
        img = cv2.imread(im, 0)        
        corners, ids, rejectedImgPts = aruco.detectMarkers(img, self.dictionary, parameters=self.parameters)
        print(len(corners), " marks detected")
        if len(corners) != 0 :
            if len(corners) < len(self.board.ids):
                for idd in self.board.ids:
                    if idd not in ids:
                        print("mark ", idd, " not detected")
                        
            if ids is not None and len(ids) > 0:
                ret, chcorners, chids = aruco.interpolateCornersCharuco(
                    corners, ids, img, self.board)
                print("{} point(s) detected".format(ret))
                print('---')
                corners_list = []
                BU = []
                for i in range (0, len(chcorners)) :
                    BU.append(chcorners[i][0])
                    corners_list.append([BU[i][0],BU[i][1],chids[i][0]])
        else :
            corners_list = False
        return (corners_list, ret) 
    
    def complete_missing_points (self, corners_list, im) :  
        """ Detection of the corners
        
        Args:
            corners_list : numpy.array
                Array of the detected (automatically) points 
            im : str
                Image path to detect
            
        Returns:
            corners_list : list (Dim = N * 3)
                List of the detected (automatically + manually) corners 
        """        

        # Find the all corners with Hessian filter                
        img_hess= plt.imread(im)
        H11, H22, H12 = hessian_matrix(img_hess, 9)
        H22 = abs(H22)
        thresh = threshold_otsu(H22)
        bin_im = H22 > thresh
        
        corners_list = np.asarray(corners_list)
        x, y, ids = np.transpose(corners_list)
        img = cv2.imread(im, 0)

        fig, ax = plt.subplots()
        plt.imshow(img, cmap='gray')
        plt.scatter(x,y, c='r')
        for name, txt in enumerate(ids):
            ax.annotate(txt, (x[name], y[name]))
            # bbox=dict(boxstyle="round", alpha=0.7, color='w')
            # ax.annotate(str(int(txt)), 
            #              xy=(x[name], y[name]),
            #              xytext=(1, 15), 
            #              textcoords='offset points',
            #              size=10,
            #              bbox=bbox)

        nx, ny = self.ncx-1, self.ncy-1
        n_corners = nx*ny
        pts_list = np.arange(n_corners)
        pts_list = np.reshape(pts_list, (ny,nx))
        pt1 = corners_list[0]
        x1, y1, id1 = pt1
        line1, column1 = np.where(pts_list==id1)
        pts_list_cut = np.delete(pts_list, line1, 0)
        pts_list_cut = np.delete(pts_list_cut, column1, 1)
        pts_list_cut = np.ravel(pts_list_cut)
        pt2 = []
        out_of_range_points = 0
        for pt in pts_list_cut :
            if np.any(corners_list[:,2] == pt) :
                line2, column2 = np.where(pts_list == pt)
                line, column = np.where(corners_list == pt)
                pt2 = corners_list[line]
                x2, y2, id2 = pt2[0]
                break
            
        if np.any(pt2) :
            # Define the referencial coordiantes of the pattern grid
            nx = line2 - line1
            ny = column2 - column1
            dx = x2 - x1
            dy = y2 - y1
            dP = math.sqrt(dx**2 + dy**2)
            l = dP / math.sqrt(nx**2 + ny**2)
            alpha = math.atan(dx/dy)
            if dy < 0 :
                alpha += math.pi
            alpha2 = math.atan(nx/ny)
            alpha1 = alpha - alpha2
            xx = l * math.sin(alpha1)
            xy = l * math.cos(alpha1)
            yx = - l * math.cos(alpha1)
            yy = l * math.sin(alpha1)

            # Define the origine point
            d0x = column1 * xx + line1 * yx
            d0y = column1 * xy + line1 * yy
            x0 = x1 - d0x
            y0 = y1 - d0y
            plt.scatter(x0,y0, c='g')
            plt.scatter(x0 + xx, y0 + xy, c='c')
            plt.scatter(x0 - yx, y0 - yy, c='c')

            # Find the holes
            for id_ in np.ravel(pts_list) :
                if np.any(corners_list[:,2] == id_) :
                    ()
                else : 
                    # Find the missing point
                    line2, column2 = np.where(pts_list == id_)
                    dix = column2 * xx + line2 * xy
                    diy = - column2 * yx - line2 * yy
                    xi = int(x0 + dix)
                    yi = int(y0 + diy)
                    d = int(l//2)

                    # Pick the missing point, if on the image
                    xm, ym = img.shape
                    if (xm < int(yi+d)) or (ym < int(xi+d)) or (0> int(yi-d)) or (0> int(xi-d)) :
                        out_of_range_points += 1
                    else :
                        # Try Hessian detection and pick the biggest binary area
                        bin_im_win = bin_im[yi-d:yi+d, xi-d:xi+d]
                        im_focus = img_hess[yi-d:yi+d, xi-d:xi+d]
                        label_img=label(clear_border(bin_im_win))
                        regions = regionprops(label_img)
                        areas = []
                        for region in (regions):
                            areas.append(region.area)
                        max_area = max(areas)
                        max_i = areas.index(max_area)
                        region = regions[max_i]
                        bary, barx = region.centroid
                        y_dot = bary + yi - d
                        x_dot = barx + xi - d

                        
                        
                        arr = np.array([x_dot, y_dot, id_])
                        plt.annotate(id_, (x_dot, y_dot))
                        plt.scatter(x_dot, y_dot, c='b')

                        corners_list = np.insert(corners_list, id_, arr, axis=0)
                        
                        # sys.exit()

                        # # If the Hessian detection is bad, manualy detection
                        # missing_points = [0, 0]
                        # def onclick(event):
                        #     global missing_points
                        #     missing_points = [event.xdata, event.ydata]
                        #     plt.close()
                        
                        # fig, ax = plt.subplots()
                        # plt.imshow(img[int(yi-d):int(yi+d), int(xi-d):int(xi+d)], cmap='gray')
                        # cid = fig.canvas.mpl_connect('button_press_event', onclick)
                        # plt.title('Click on the missing corner')
                        # plt.show()
                        # plt.waitforbuttonpress()
                        # xi = xi+missing_points[0]-d
                        # yi = yi+missing_points[1]-d
                        # fig, ax = plt.subplots()
                        # plt.imshow(img[int(yi-10):int(yi+10), int(xi-10):int(xi+10)], cmap='gray')
                        # cid = fig.canvas.mpl_connect('button_press_event', onclick)
                        # plt.title('Click again')
                        # plt.show()
                        # plt.waitforbuttonpress()
                        # xi = xi+missing_points[0]-d
                        # yi = yi+missing_points[1]-d
                        # arr = np.array([xi, yi, id_])
                        # corners_list = np.insert(corners_list, id_, arr, axis=0)
                        
        else :
            print('Impossible to detect manualy corners of image : ', im)
            corners_list = False
        print (out_of_range_points, ' points out of the image or to close to the border')
        return (corners_list)

   
def calibration_model(nx, ny, l) : 
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

def cut_calibration_model (List_images, 
                           Xref, 
                           __dict__) :
    """ Group all of the images detected and filter the points not detected. 
        For each corners not detected on an image, delete it on all the others images. 
        Delete also on the real positions of the corners.
    
    Args:
        List_images : list
            List of the detected corners
        Xref : list
            List of the real corners
        pattern : str
            Name of the pattern used ('macro' or 'micro')
        
    Returns:
        all_x : list (Dim = Nimages * N * 3)
            List of the real corners
        all_X : list (Dim = Nimages * N * 3)
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
        B, pts = Calibrate(__dict__).calibrate(sorted(glob(List_images[i]))[0])
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



def NAN_calibration_model (List_images, 
                           Xref, 
                           __dict__,
                           hybrid_detection = True) :
    """ Group all of the images detected and filter the points not detected. 
        For each corners not detected on an image, replace the points with NAN. 
    
    Args:
        List_images : list
            List of the detected corners
        Xref : list
            List of the real corners
        pattern : str
            Name of the pattern used ('macro' or 'micro')
        hybrid_detection : bool, optional
            If True, then the missing points are filled with manual/visual 
            detection
        
    Returns:
        all_x : np.array (Dim = Nimages * N * 3)
            Array of the real corners
        all_X : np.array (Dim = Nimages * N * 3)
            Array of the detected corners
            
    """    
    M = len(List_images)
    
    # First, detect the holes = missing points
    Nall = len(Xref)
    nb_pts = np.zeros(M)
    all_X = np.zeros((M, Nall, 3))
    for i in range (0, M) :
        im = sorted(glob(List_images[i]))[0]
        corners_list, pts = Calibrate(__dict__).calibrate(im)
        nb_pts[i] = pts
        corners_list = np.asarray(corners_list)
        if hybrid_detection :
            corners_list = Calibrate(__dict__).complete_missing_points(corners_list, im)
            
        build = 0
        while build < Nall :
            if len(corners_list) == build :
                corners_list = np.insert (corners_list, 
                                          build, 
                                          [np.nan, np.nan, build], 
                                          axis = 0)
                build += 1
            else :
                if corners_list[build, 2] == build :
                    build += 1
                else :
                    corners_list = np.insert (corners_list, 
                                              build, 
                                              [np.nan, np.nan, build], 
                                              axis = 0)
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






def pattern_detection (__dict__,
                       detection = True,
                       NAN = False,
                       saving_folder = 'Folders_npy') :
    """Detect the corners of Charucco's pattern.
    
    Args:
       __dict__ : dict
           Pattern properties define in a dict.
       detection : bool, optional
           If True, all the analysis will be done. If False, the code will take the informations in 'saving_folder'
       saving_folder : str, optional
           Folder to save datas
           
    Returns:
       all_X : numpy.ndarray
           The corners of the pattern detect by the camera ranged in an array arrange with all left pictures followed by all right pictures. 
           Expl : [left_picture_1, left_picture_2, right_picture_1, right_picture_2]
       all_x : numpy.ndarray
           The theorical corners of the pattern
    """
    # Taking the main parameters from bibliotheque_data_eddy.
    left_folder = __dict__['left_folder']
    right_folder = __dict__['right_folder']
    name = __dict__['name']
    ncx = __dict__['ncx']
    ncy = __dict__['ncy']
    sqr = __dict__['sqr']
    Images_left = sorted(glob(str(left_folder) + '/*'))
    Images_right = sorted(glob(str(right_folder) + '/*'))
    Images = Images_left
    for i in range (len(Images_right)) :
        Images.append(Images_right[i])
    
    Save_Ucam_Xref = [str(saving_folder) + "/all_X_" + name + ".npy", 
                      str(saving_folder) + "/all_x_" + name + ".npy", 
                      str(saving_folder) + "/nb_pts_" + name + ".npy"]
    
    # Corners detection
    if detection :
        print('    - Detection of the pattern in progress ...')
        # Creation of the theoretical pattern + detection of camera's pattern
        Xref = calibration_model(ncx, ncy, sqr)
        if NAN :
            all_x, all_X, nb_pts = NAN_calibration_model(Images, Xref, __dict__)
        else :
            all_x, all_X, nb_pts = cut_calibration_model(Images, Xref, __dict__)
        
        if not np.any(all_X[0]):
            print('Not any point detected in all images/cameras')
        else :
            np.save(Save_Ucam_Xref[0], all_X)
            np.save(Save_Ucam_Xref[1], all_x)
            np.save(Save_Ucam_Xref[2], nb_pts)
    
            print('    - Saving datas in ', saving_folder)
    # Taking pre-calculated datas from the saving_folder
    else :
        print('    - Taking datas from ', saving_folder)        
        all_X = np.load(Save_Ucam_Xref[0])
        all_x = np.load(Save_Ucam_Xref[1])
        nb_pts = np.load(Save_Ucam_Xref[2])
        print(Save_Ucam_Xref[2])
        
    return(all_X, all_x, nb_pts)



def DIC_3D_detection (__DIC_dict__,
                      detection = True,
                      saving_folder = 'Folders_npy',
                      flip = False) :
    """Use the DIC to locate all the points from the left picture in the right
    twin picture.
    
    Args:
       __DIC_dict__ : dict
           DIC dictionnary including the picture folders, the saving name and 
           the window (in px) to study.
       detection : bool, optional
           If True, all the analysis will be done. If False, the code will 
           take the informations in 'saving_folder'
       saving_folder : str, optional
           Folder to save or take datas
       flip : bool, optional
           If True, all the pictures are flipped before the DIC (useful when 
           you're using a mirror)
           
    Returns:
       all_X : numpy.ndarray
           All the points of the left pictures (1 point per pixel) in an array 
           arrange with their positions in their right twin picture. 
           Expl : all_X = [left_picture_1, 
                           left_picture_2, 
                           right_picture_1 (Compared with left_picture_1), 
                           right_picture_2 (Compared with left_picture_2)]
    """
    left_folder = __DIC_dict__['left_folder']
    right_folder = __DIC_dict__['right_folder']
    name = __DIC_dict__['name']
    window = __DIC_dict__['window']
    Save_all_U = str(saving_folder) +"/all_U_" + name + ".npy"
    Save_all_V = str(saving_folder) +"/all_V_" + name + ".npy"
    Save_X_map = str(saving_folder) +"/X_map_" + name + ".npy"
    
    Images_left = sorted(glob(str(left_folder) + '/*.tif'))
    Images_right = sorted(glob(str(right_folder) + '/*.tif'))
    Images = Images_left
    N = len(Images)
    for i in range (N) :
        Images.append(Images_right[i]) 
    [lx1, lx2], [ly1, ly2] = window
    all_left = []
    all_right = []

    # Corners detection
    print('    - DIC in progress ...')
    # DIC detection of the points from each camera
    for i in range (N) :
        if detection :
            image_1, image_2 = Images[i], Images[i+N]
            U, V = DIC.strain_field(image_1, 
                                    image_2, 
                                    flip = flip)
            if i == 0 :
                all_U = np.empty((N, U.shape[0], U.shape[1]))
                all_V = np.empty((N, V.shape[0], V.shape[1]))
            all_U[i] = U
            all_V[i] = V
            np.save(Save_all_U, all_U)
            np.save(Save_all_V, all_V)
            print('    - Saving datas in ', saving_folder)
        else :
            # Taking pre-calculated datas from the saving_folder
            print('    - Taking datas from ', saving_folder)        
            all_U = np.load(Save_all_U)
            all_V = np.load(Save_all_V)
            X_map = np.load(Save_X_map)
        
    for i in range (N) :
        U, V = all_U[i], all_V[i]
        nX1, nX2 = U.shape
        # ntot = (lx2 - lx1) * (ly2 - ly1)
        linsp = np.arange(nX1)+1
        linsp = np.reshape (linsp, (1,nX1))
        X1matrix = np.matmul(np.ones((nX1, 1)), linsp)
        X2matrix = np.matmul(np.transpose(linsp), np.ones((1, nX1)))
        X1matrix_w = X1matrix[ly1:ly2, lx1:lx2]
        X2matrix_w = X2matrix[ly1:ly2, lx1:lx2]

        if detection :
            # Generate the mapping
            X_map = np.transpose(np.array([np.ravel(X1matrix), np.ravel(X2matrix)]))
            X_map = X_map.reshape((X1matrix.shape[0],X1matrix.shape[1],2))
            np.save(Save_X_map, X_map)

        # Left camera --> position = each px
        X_c1 = np.transpose(np.array([np.ravel(X1matrix_w), np.ravel(X2matrix_w)]))
        UV = np.transpose(np.array([np.ravel(U[ly1:ly2, lx1:lx2]), np.ravel(V[ly1:ly2, lx1:lx2])]))

        # Right camera --> position = each px + displacement
        X_c2 = X_c1 + UV

        all_left.append(X_c1)
        all_right.append(X_c2)
    all_X = all_left
    for i in range (N) :
        all_X.append(all_right[i])
    all_X = np.asarray(all_X)
    
    return(all_X)



def DIC_3D_detection_lagrangian (__DIC_dict__,
                      detection = True,
                      saving_folder = 'Folders_npy',
                      flip = False,
                      img_ref = '') :
    """Use the DIC to locate all the points from the reference picture
    (first left one) in the deformed ones (other left and right pictures).
    
    Args:
       __DIC_dict__ : dict
           DIC dictionnary including the picture folders, the saving name and 
           the window (in px) to study.
       detection : bool, optional
           If True, all the analysis will be done. If False, the code will 
           take the informations in 'saving_folder'
       saving_folder : str, optional
           Folder to save or take datas
       flip : bool, optional
           If True, all the pictures are flipped before the DIC (useful when 
           you're using a mirror)
       img_ref : str, optional
           Name of the reference image for the DIC calculation
           
    Returns:
       all_X : numpy.ndarray
           All the points of the first picture (1 point per pixel) in an array 
           arrange with their positions in all left pictures followed by all 
           right pictures. 
           Expl : all_X = [left_picture_1 = ref picture, 
                           left_picture_2, 
                           right_picture_1, 
                           right_picture_2]
    """
    left_folder = __DIC_dict__['left_folder']
    right_folder = __DIC_dict__['right_folder']
    name = __DIC_dict__['name']
    window = __DIC_dict__['window']
    Save_all_U = str(saving_folder) +"/Lagrangian_all_U_" + name + ".npy"
    Save_all_V = str(saving_folder) +"/Lagrangian_all_V_" + name + ".npy"
    Save_X_map = str(saving_folder) +"/X_map_" + name + ".npy"
    
    Images_left = sorted(glob(str(left_folder) + '/*.tif'))
    Images_right = sorted(glob(str(right_folder) + '/*.tif'))
    Images = Images_left
    N = len(Images)
    for i in range (N) :
        Images.append(Images_right[i]) 
    [lx1, lx2], [ly1, ly2] = window
    N = len(Images)

    # Corners detection
    print('    - DIC in progress ...')
    # DIC detection of the points from each camera
    for img in Images :
        if img == img_ref:
            image_ref = img_ref
        else :
            image_ref = Images[0]
    print('Image reference = ', image_ref)

    all_X = []
    for i in range (N) :
        if detection :
            image_def = Images[i]
            U, V = DIC.strain_field(image_ref, 
                                    image_def, 
                                    flip = flip)
            if i == 0 :
                all_U = np.empty((N, U.shape[0], U.shape[1]))
                all_V = np.empty((N, V.shape[0], V.shape[1]))
            all_U[i] = U
            all_V[i] = V
            np.save(Save_all_U, all_U)
            np.save(Save_all_V, all_V)
            print('    - Saving datas in ', saving_folder)
        else :
            # Taking pre-calculated datas from the saving_folder
            print('    - Taking datas from ', saving_folder)        
            all_U = np.load(Save_all_U)
            all_V = np.load(Save_all_V)
            X_map = np.load(Save_X_map)
        
    for i in range (N) :
        U, V = all_U[i], all_V[i]
        nX1, nX2 = U.shape
        # ntot = (lx2 - lx1) * (ly2 - ly1)
        linsp = np.arange(nX1)+1
        linsp = np.reshape (linsp, (1,nX1))
        X1matrix = np.matmul(np.ones((nX1, 1)), linsp)
        X2matrix = np.matmul(np.transpose(linsp), np.ones((1, nX1)))
        X1matrix_w = X1matrix[ly1:ly2, lx1:lx2]
        X2matrix_w = X2matrix[ly1:ly2, lx1:lx2]

        if detection :
            # Generate the mapping
            X_map = np.transpose(np.array([np.ravel(X1matrix), np.ravel(X2matrix)]))
            X_map = X_map.reshape((X1matrix.shape[0],X1matrix.shape[1],2))
            np.save(Save_X_map, X_map)

        # Left camera --> position = each px
        X_c1 = np.transpose(np.array([np.ravel(X1matrix_w), np.ravel(X2matrix_w)]))
        UV = np.transpose(np.array([np.ravel(U[ly1:ly2, lx1:lx2]), np.ravel(V[ly1:ly2, lx1:lx2])]))

        # Right camera --> position = each px + displacement
        X_c2 = X_c1 + UV
        all_X.append(X_c2)
    all_X = np.array(all_X)
    return(all_X)



def DIC_fields (__DIC_dict__,
                detection = True,
                saving_folder = 'Folders_npy',
                flip = False) :
    """Use the DIC to calcul all the left deformed fields ans the right 
    deformed fields.
    
    Args:
       __DIC_dict__ : dict
           DIC dictionnary including the picture folders, the saving name and 
           the window (in px) to study.
       detection : bool, optional
           If True, all the analysis will be done. If False, the code will 
           take the informations in 'saving_folder'
       saving_folder : str, optional
           Folder to save or take datas
       flip : bool, optional
           If True, all the pictures are flipped before the DIC (useful when 
           you're using a mirror)
           
    Returns:
       U_left : numpy.ndarray
           All the left deformed fields in x direction.
       V_left : numpy.ndarray 
           All the left deformed fields in y direction.
       U_right : numpy.ndarray 
           All the right deformed fields in x direction.
       V_right : numpy.ndarray
           All the right deformed fields in y direction.
    """
    left_folder = __DIC_dict__['left_folder']
    right_folder = __DIC_dict__['right_folder']
    name = __DIC_dict__['name']
    Save_UV = str(saving_folder) +"/all_UV_" + name + ".npy"
    if detection :
        Images_left = sorted(glob(str(left_folder) + '/*.tif'))
        Images_right = sorted(glob(str(right_folder) + '/*.tif'))
        Images = Images_left
        N = len(Images)
        for i in range (N) :
            Images.append(Images_right[i])    
        # Corners detection
        if detection :
            print('    - DIC in progress ...')
            # DIC detection of the points from each camera
            for i in range (N) :
                image_t0_left, image_ti_left = Images[0], Images[i]
                image_t0_right, image_ti_right = Images[N], Images[i+N]
                Ul, Vl = DIC.strain_field(image_t0_left, 
                                          image_ti_left,
                                          flip = flip)
                Ur, Vr = DIC.strain_field(image_t0_right, 
                                          image_ti_right,
                                          flip = flip)
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
    # Taking pre-calculated datas from the saving_folder
    else :
        print('    - Taking datas from ', saving_folder)        
        all_UV = np.load(Save_UV)
        U_left, V_left, U_right, V_right = all_UV
        
    return(U_left, V_left, U_right, V_right)



def camera_np_coordinates (all_X, 
                           all_x, 
                           x3_list) :
    """Organising the coordinates of the calibration
    
    Args:
       all_X : numpy.ndarray
           The corners of the pattern detect by the camera
       all_x : numpy.ndarray
           The theorical corners of the pattern
       x3_list : numpy.ndarray
           List of the different z position. (Ordered the same way in the target folder)
       saving_folder : str, optional
           Where to save datas
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
            all_xi[j][:,2] = x3_list[j]

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
    # If there is some NAN value, then delete all 2D and 3D corresponding points
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


if __name__ == '__main__' :
    main_path = '/home/caroneddy/These/Stereo_camera/Pycaso_archives/src'    
    
    # Define the inputs
    __calibration_dict__ = {
    'left_folder' : main_path + '/Images_example/2022_02_28/test1',
    'right_folder' : main_path + '/Images_example/2022_02_28/test2',
    'name' : 'micro_calibration',
    'ncx' : 16,
    'ncy' : 12,
    'sqr' : 0.3}
    
    # __DIC_dict__ = {
    # 'left_folder' : main_path + '/Images_example/2022_02_28/left_coin_identification',
    # 'right_folder' : main_path + '/Images_example/2022_02_28/right_coin_identification',
    # 'name' : 'micro_identification',
    # 'window' : [[300, 1700], [300, 1700]]}
    
    # Create the list of z plans
    Folder = __calibration_dict__['left_folder']
    Imgs = sorted(glob(str(Folder) + '/*'))
    x3_list = np.zeros((len(Imgs)))
    for i in range (len(Imgs)) :
        x3_list[i] = float(Imgs[i][len(Folder)+ 1:-4])

    # Chose the degrees for Soloff and direct polynomial fitting
    polynomial_form = 332
    direct_polynomial_form = 4

    # Create the result folder if not exist
    saving_folder = main_path + '/results/2022_02_28_results/Test'
    
    # Create the result folder if not exist
    if os.path.exists(saving_folder) :
        ()
    else :
        P = pathlib.Path(saving_folder)
        pathlib.Path.mkdir(P, parents = True)    
    
    
    all_X, all_x, nb_pts = pattern_detection(__calibration_dict__,
                                            detection = True,
                                            NAN = True,
                                            saving_folder = saving_folder)    
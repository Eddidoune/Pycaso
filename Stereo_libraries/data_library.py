#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
from glob import glob
from copy import deepcopy

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
    
    def calibrate(self, imgs):
        """ Detection of the corners
        
        Args:
            imgs : list
                List of the images paths to detect
            
        Returns:
            corners_list : list (Dim = N * 3)
                List of the detected corners 
        """
        for im in imgs:
            print("=> Calculation of the image {0}".format(im))
            img = cv2.imread(im, 0)
            corners, ids, rejectedImgPts = aruco.detectMarkers(img, self.dictionary, parameters=self.parameters)
            print(len(corners), " marks detected")
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

def cut_calibration_model (List_images, Xcal, __dict__) :
    """ Group all of the images detected and filter the points not detected. 
        For each corners not detected on an image, delete it on all the others images. 
        Delete also on the real positions of the corners.
    
    Args:
        List_images : list
            List of the detected corners
        Xcal : list
            List of the real corners
        pattern : str
            Name of the pattern used ('macro' or 'micro')
        
    Returns:
        all_Xref : list (Dim = Nimages * N * 3)
            List of the real corners
        all_Ucam : list (Dim = Nimages * N * 3)
            List of the detected corners
            
    """    
    Ucam_init = []
    Xref = []
    holes = [] # List of the missing points
    M = len(List_images)
    
    # First, detect the holes = missing points
    for i in range (0, M) :
        B = Calibrate(__dict__).calibrate(sorted(glob(List_images[i])))
        Ucam_init.append(B)
        N = len(B)
        points = []
        for j in range (0, N) :
            points.append(B[j][2])
        Nall = len(Xcal)
        holes_j = [j for j in range(Nall)]
        for j in range (0, N) :
            p = points[N-(j+1)]
            del(holes_j[p])
        holes = holes + holes_j 
    
    # Then arrange the holes and delect twins
    holes = list(dict.fromkeys(holes))
    holes = sorted(holes)
    T = len(holes)
    
    # Then delete those holes on all_Ucam
    all_Ucam = []
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
        all_Ucam.append(Ucam_remove)
        
    Pmax = len(Xcal)
    print('----------')
    print(str(T) + ' points deleted in each images on a total of ' + str(Pmax) + ' points')
    print('----------')
          
    # Then delete those holes on all_Xref
    Xref = deepcopy(Xcal)
    for t in range (0, T) :     
        p = holes[T-(t+1)]
        del(Xref[p])
    all_Xref = []
    for i in range (0, M) :
        all_Xref.append(Xref)
    return (all_Xref, all_Ucam)


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
           Sabing folder to save datas
           
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
        Xref = calibration_model(ncx, ncy, sqr)
        all_Xref, all_Ucam = cut_calibration_model(Images, Xref, __dict__)
        
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
    return (xc1, xc2, Xc1, Xc2)


if __name__ == '__main__' :
    ()
    # # Show the reals and theoreticals points  
    # micro = {
    # 'name' : 'micro',
    # 'ncx' : 16,
    # 'ncy' : 12 ,
    # 'sqr' : 300*(10**(-6))}  
    # macro = {
    # 'name' : 'macro',
    # 'ncx' : 10,
    # 'ncy' : 8 ,
    # 'sqr' : 7.35*(10**(-3))}  
    # test = {
    # 'name' : 'test',
    # 'ncx' : 20,
    # 'ncy' : 20 ,
    # 'sqr' : 7.35*(10**(-3))}  
    # # Choose the dict
    # __dict__ = micro
    
    # Images = sorted(glob('./Images/micro/right-test/*.tif'))
    # ncx = __dict__['ncx']
    # ncy = __dict__['ncy']
    # sqr = __dict__['sqr']
    # mrk = sqr/2
    
    # # Corners detection
    # print('    - Detection of the pattern in progress ...')
    # # Creation of the theoretical pattern + detection of camera's pattern
    # Xref = calibration_model(ncx, ncy, sqr)
    # all_Xref, all_Ucam = cut_calibration_model(Images, Xref, __dict__)

    # all_Xref = np.asarray(all_Xref)
    # all_Ucam = np.asarray(all_Ucam)

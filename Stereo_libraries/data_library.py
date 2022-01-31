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
            BA : list (Dim = N * 3)
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
                BA = []
                BU = []
                for i in range (0, len(chcorners)) :
                    BU.append(chcorners[i][0])
                    BA.append([BU[i][0],BU[i][1],chids[i][0]])
        return (BA) 

    def ChArucco_board (self, dpi = 300) :
        """Create ChArucco board for the calibration
        
        Args:
           dpi : int
               Resolution of the picture
           
        Returns:
           ChArucco_Board_build_img : numpy.ndarray
               Black and white array --> Save as a png
        """
        ncx = self.ncx
        ncy = self.ncy
        sqr = self.sqr
        mrk = self.mrk
        n = ncx * ncy / 2
        ChArucco_Board_img = []
        r = sqr//mrk
        if r != 2 :
            print('Impossible to print the ChArucco board (unresolve for sqr/mrk = 2) ')
            ChArucco_Board_img = None
        for e in range (int(n)) :
            ChArucco_mrk = cv2.aruco.drawMarker(self.dictionary, e, 100)
            x, y = ChArucco_mrk.shape
            dx, dy = x//2, y//2
            black_square = np.ones ((x//2,y//2)) * 255
            ChArucco_square = np.empty((2*x, 2*y))
            for i in range (4) :
                for j in range (4) :
                    if (i == 0) or (i == 3) or (j == 0) or (j == 3) :
                        ChArucco_square[dx*i:dx*(i+1),dy*j:dy*(j+1)] = black_square
                    else :
                        ChArucco_square[dx*i:dx*(i+1),dy*j:dy*(j+1)] = ChArucco_mrk[dx*(i-1):dx*i,dy*(j-1):dy*j]          
            ChArucco_Board_img.append(ChArucco_square)
        x, y = ChArucco_Board_img[0].shape
        white_square = np.zeros ((x,y))
        shapetot = (y*ncy, x*ncx)
        ChArucco_Board_build_img = np.empty(shapetot)
        e = 0
        for i in range(ncy) :
            for j in range(ncx) :
                if ((i+j)%2) == 0 :
                    ChArucco_Board_build_img[x*i:x*(i+1),y*j:y*(j+1)] = ChArucco_Board_img[e]
                    e += 1
                else :
                    ChArucco_Board_build_img[x*i:x*(i+1),y*j:y*(j+1)] = white_square
        plt.imshow(ChArucco_Board_build_img, interpolation='nearest', cmap='gray', vmin=0, vmax=255)
        plt.savefig('ChArucco_Board_build_img.png', dpi=dpi)
        plt.show()        
        return (ChArucco_Board_build_img)
   
def Modele_calibration(nx, ny, l) : 
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

def Modeles_calibration_images_tronquage (List_images, Xcal, __dict__) :
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
               saving_folder = 'Fichiers_txt') :
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
        Xref = Modele_calibration(ncx, ncy, sqr)
        all_Xref, all_Ucam = Modeles_calibration_images_tronquage(Images, Xref, __dict__)
        
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
        all_Ucam = np.load(str(saving_folder) +"/all_Ucam_" + name + ".npy")
        all_Xref = np.load(str(saving_folder) +"/all_Xref_" + name + ".npy")
    return(all_Ucam, all_Xref)


if __name__ == '__main__' :
    # Show the reals and theoreticals points  
    micro = {
    'name' : 'micro',
    'ncx' : 16,
    'ncy' : 12 ,
    'sqr' : 300*(10**(-6))}  
    macro = {
    'name' : 'macro',
    'ncx' : 10,
    'ncy' : 8 ,
    'sqr' : 7.35*(10**(-3))}  
    test = {
    'name' : 'test',
    'ncx' : 20,
    'ncy' : 20 ,
    'sqr' : 7.35*(10**(-3))}  
    # Choose the dict
    __dict__ = micro
    
    Images = sorted(glob('./Images/micro/right-test/*.tif'))
    ncx = __dict__['ncx']
    ncy = __dict__['ncy']
    sqr = __dict__['sqr']
    mrk = sqr/2
    
    # Corners detection
    print('    - Detection of the pattern in progress ...')
    # Creation of the theoretical pattern + detection of camera's pattern
    Xref = Modele_calibration(ncx, ncy, sqr)
    all_Xref, all_Ucam = Modeles_calibration_images_tronquage(Images, Xref, __dict__)

    all_Xref = np.asarray(all_Xref)
    all_Ucam = np.asarray(all_Ucam)

    # ChArucco_Board_build_img = Calibrate (__dict__).ChArucco_board()    

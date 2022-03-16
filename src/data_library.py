#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from glob import glob
from copy import deepcopy
import DIC

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
        all_x : list (Dim = Nimages * N * 3)
            List of the real corners
        all_X : list (Dim = Nimages * N * 3)
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
        
    Pmax = len(Xcal)
    print('----------')
    print(str(T) + ' points deleted in each images on a total of ' + str(Pmax) + ' points')
    print('----------')
          
    # Then delete those holes on all_x
    Xref = deepcopy(Xcal)
    for t in range (0, T) :     
        p = holes[T-(t+1)]
        del(Xref[p])
    all_x = []
    for i in range (0, M) :
        all_x.append(Xref)
    return (all_x, all_X)


def pattern_detection (__dict__,
               detection = True,
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
    
    Save_Ucam_Xref = [str(saving_folder) +"/all_X_" + name + ".npy", str(saving_folder) + "/all_x_" + name + ".npy"]
    
    # Corners detection
    if detection :
        print('    - Detection of the pattern in progress ...')
        # Creation of the theoretical pattern + detection of camera's pattern
        Xref = calibration_model(ncx, ncy, sqr)
        all_x, all_X = cut_calibration_model(Images, Xref, __dict__)

        all_x = np.asarray(all_x)
        all_x = all_x[:, :, [0, 1]]
        all_X = np.asarray(all_X)
        all_X = all_X[:, :, [0, 1]]

        np.save(Save_Ucam_Xref[0], all_X)
        np.save(Save_Ucam_Xref[1], all_x)

        print('    - Saving datas in ', saving_folder)
    # Taking pre-calculated datas from the saving_folder
    else :
        print('    - Taking datas from ', saving_folder)        
        all_X = np.load(Save_Ucam_Xref[0])
        all_x = np.load(Save_Ucam_Xref[1])
    return(all_X, all_x)



def DIC_detection (__dict__,
                   detection = True,
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
    """
    left_folder = __dict__['left_folder']
    right_folder = __dict__['right_folder']
    name = __dict__['name']
    window = __dict__['window']
    Images_left = sorted(glob(str(left_folder) + '/*.tif'))
    Images_right = sorted(glob(str(right_folder) + '/*.tif'))
    Images = Images_left
    N = len(Images)
    for i in range (N) :
        Images.append(Images_right[i]) 
    
    Save_Ucam_Xref = str(saving_folder) +"/all_X_DIC_" + name + ".npy"
    [lx1, lx2], [ly1, ly2] = window
    all_left = []
    all_right = []

    # Corners detection
    if detection :
        print('    - DIC in progress ...')
        # DIC detection of the points from each camera
        for i in range (N) :
            image_1, image_2 = Images[i], Images[i+N]
            U, V = DIC.strain_field(image_1, image_2)

            nX1, nX2 = U.shape
            ntot = (lx2 - lx1) * (ly2 - ly1)
            linsp = np.arange(nX1)+1
            linsp = np.reshape (linsp, (1,nX1))
            X1matrix = np.matmul(np.ones((nX1, 1)), linsp)
            X2matrix = np.matmul(np.transpose(linsp), np.ones((1, nX1)))
            X1matrix = X1matrix[ly1:ly2, lx1:lx2]
            X2matrix = X2matrix[ly1:ly2, lx1:lx2]

            # Left camera --> position = each px
            X_c1 = np.transpose(np.array([X1matrix.reshape(ntot), X2matrix.reshape(ntot)]))
            UV = np.transpose(np.array([U[ly1:ly2, lx1:lx2].reshape(ntot), V[ly1:ly2, lx1:lx2].reshape(ntot)]))

            # Right camera --> position = each px + displacement
            X_c2 = X_c1 + UV

            all_left.append(X_c1)
            all_right.append(X_c2)
        all_X = all_left
        for i in range (N) :
            all_X.append(all_right[i])
        all_X = np.asarray(all_X)
        np.save(Save_Ucam_Xref, all_X)
        print('    - Saving datas in ', saving_folder)
    # Taking pre-calculated datas from the saving_folder
    else :
        print('    - Taking datas from ', saving_folder)        
        all_X = np.load(Save_Ucam_Xref)
    
    return(all_X)



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
    return (x, Xc1, Xc2)


if __name__ == '__main__' :
    ()
    # # Show the reals and theoreticals points  
    # micro = {
    # 'name' : 'micro',
    # 'ncx' : 16,
    # 'ncy' : 12 ,
    # 'sqr' : 0.3}  
    # macro = {
    # 'name' : 'macro',
    # 'ncx' : 10,
    # 'ncy' : 8 ,
    # 'sqr' : 7.35}  
    # test = {
    # 'name' : 'test',
    # 'ncx' : 20,
    # 'ncy' : 20 ,
    # 'sqr' : 7.35}  
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
    # all_x, all_X = cut_calibration_model(Images, Xref, __dict__)

    # all_x = np.asarray(all_x)
    # all_X = np.asarray(all_X)

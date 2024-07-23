#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 09:19:07 2021

@author: Eddidoune
"""
try : 
    import cupy as np
except ImportError:
    import numpy as np
import sys
import pathlib
import os
from glob import glob

sys.path.append('../pycaso')

import pycaso as pcs
import data_library as data
import matplotlib.pyplot as plt

if __name__ == '__main__' :    
    saving_folder = 'results/main_exemple'
    # Define the inputs
    calibration_dict = {
    'cam1_folder' : 'Images_example/left_calibration11',
    'cam2_folder' : 'Images_example/right_calibration11',
    'name' : 'micro_calibration',
    'saving_folder' : saving_folder,
    'ncx' : 16,
    'ncy' : 12,
    'sqr' : 0.3}
    
    DIC_dict = {
    'cam1_folder' : 'Images_example/left_identification',
    'cam2_folder' : 'Images_example/right_identification',
    'name' : 'micro_identification',
    'saving_folder' : saving_folder,
    'window' : [[300, 1700], [300, 1700]]}
    
    # Create the list of z plans
    Folder = calibration_dict['cam1_folder']
    Imgs = sorted(glob(str(Folder) + '/*'))
    z_list = np.zeros((len(Imgs)))
    for i in range (len(Imgs)) :
        z_list[i] = float(Imgs[i][len(Folder)+ 1:-4])
    
    # Chose the degrees for Soloff, direct and Zernike polynomial fitting
    Soloff_pform = 332
    direct_pform = 4
    Zernike_pform = 4
    
    # Create the result folder if not exist
    if os.path.exists(saving_folder) :
        ()
    else :
        P = pathlib.Path(saving_folder)
        pathlib.Path.mkdir(P, parents = True)
    
    print('')
    print('#####       ')
    print('Direct method - Start calibration')
    print('#####       ')
    print('')
    direct_constants, Mag = pcs.direct_calibration (z_list,
                                                    direct_pform,
                                                    **calibration_dict)
    sys.exit()
    print('')
    print('#####       ')
    print('Zernike method - Start calibration')
    print('#####       ')  
    Zernike_constants, Magnification = pcs.Zernike_calibration (z_list,
                                                                Zernike_pform,
                                                                **calibration_dict)

    print('')
    print('#####       ')
    print('Soloff method - Start calibration')
    print('#####       ')  
    Soloff_constants0, Soloff_constants, Mag = pcs.Soloff_calibration (z_list,
                                                                       Soloff_pform,
                                                                       **calibration_dict)
    
    print('')
    print('#####       ')
    print('Identification of displacements field by DIC')
    print('#####       ')
    print('')
    Xcam1_id, Xcam2_id = data.DIC_get_positions(DIC_dict)
    
    print('')
    print('#####       ')
    print('Calculation of 3D view')
    print('#####       ')
    print('')   
    # Direct identification
    xDirect_solution = pcs.direct_identification (Xcam1_id[0],
                                                  Xcam2_id[0],
                                                  direct_constants,
                                                  direct_pform)
    xD, yD, zD = xDirect_solution
    wnd = DIC_dict['window']    
    zD = zD.reshape((wnd[0][1] - wnd[0][0], wnd[1][1] - wnd[1][0]))
    
    plt.figure()
    plt.imshow(zD)
    plt.title('Z projected on left camera with direct calculation')
    cb = plt.colorbar()
    plt.clim(2.6, 3)
    cb.set_label('z in mm')
    plt.show() 
    
    # Zernike identification
    # Identify the calibration dimensions
    Cameras_dimensions = data.cameras_size(**calibration_dict)
    xZernike_solution = pcs.Zernike_identification (Xcam1_id[0],
                                                    Xcam2_id[0],
                                                    Zernike_constants,
                                                    Zernike_pform,
                                                    Cameras_dimensions)
    xZ, yZ, zZ = xZernike_solution
    wnd = DIC_dict['window']    
    zZ = zZ.reshape((wnd[0][1] - wnd[0][0], wnd[1][1] - wnd[1][0]))
    
    plt.figure()
    plt.imshow(zZ)
    plt.title('Z projected on left camera with Zernike calculation')
    cb = plt.colorbar()
    plt.clim(2.6, 3)
    cb.set_label('z in mm')
    plt.show() 
    
    # Soloff identification
    soloff_file = saving_folder + '/xsolution_soloff0.npy'
    import time
    t00 = time.time()
    
    # Condition not to calculate Soloff points if already exists in a
    # database
    if os.path.exists(soloff_file) :
        xSoloff_solution = np.load(soloff_file)
    else :
        xSoloff_solution = pcs.Soloff_identification (Xcam1_id[0],
                                                      Xcam2_id[0],
                                                      Soloff_constants0, 
                                                      Soloff_constants,
                                                      Soloff_pform,
                                                      method = 'curve_fit')
        np.save(soloff_file, xSoloff_solution)
    
    xS, yS, zS = xSoloff_solution
    xS = xS.reshape((wnd[0][1] - wnd[0][0], wnd[1][1] - wnd[1][0]))
    yS = yS.reshape((wnd[0][1] - wnd[0][0], wnd[1][1] - wnd[1][0]))
    zS = zS.reshape((wnd[0][1] - wnd[0][0], wnd[1][1] - wnd[1][0]))
    t0 = time.time()
    print('tS : ',t0-t00)
      
    plt.figure()
    plt.imshow(zS)
    plt.title('Z projected on left camera with Soloff calculation')
    cb = plt.colorbar()
    plt.clim(2.6, 3)
    cb.set_label('z in mm')
    plt.show()
    
    AI_training_size = 50000
    AIfile = saving_folder + '/Soloff_AI_training.csv'
    model = pcs.AI_training (Xcam1_id[0],
                             Xcam2_id[0],
                             xSoloff_solution,
                             AI_training_size = AI_training_size,
                             file = AIfile)
    
    xAI_solution = pcs.AI_identification (Xcam1_id[0],
                                          Xcam2_id[0],
                                          model)

    xAI, yAI, zAI = xAI_solution
      
    plt.figure()
    plt.imshow(zAI)
    plt.title('Z projected on left camera with AI calculation')
    cb = plt.colorbar()
    plt.clim(2.6, 3)
    cb.set_label('z in mm')
    plt.show()
    t1 = time.time()
    print('tAI : ',t1-t0)
    
    xdiff = xAI - xS
    ydiff = yAI - yS
    zdiff = zAI - zS
    rdiff = np.sqrt(xdiff**2 + ydiff**2 + zdiff**2)
    print('max : ', np.max(np.abs(xdiff)), np.max(np.abs(ydiff)), np.max(np.abs(zdiff)), np.max(np.abs(rdiff)))
    print('mean : ', np.mean(np.abs(xdiff)), np.mean(np.abs(ydiff)), np.mean(np.abs(zdiff)), np.mean(np.abs(rdiff)))
    print('std : ', np.std(xdiff), np.std(ydiff), np.std(zdiff), np.std(rdiff))

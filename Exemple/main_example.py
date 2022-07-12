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
sys.path.append('../Pycaso')

import Pycaso as pcs
import data_library as data
import matplotlib.pyplot as plt

if __name__ == '__main__' :    
    saving_folder = 'results/main_expl_300_1700'
    # Define the inputs
    __calibration_dict__ = {
    'left_folder' : 'Images_example/left_calibration',
    'right_folder' : 'Images_example/right_calibration',
    'name' : 'micro_calibration',
    'saving_folder' : saving_folder,
    'ncx' : 16,
    'ncy' : 12,
    'sqr' : 0.3}
    
    __DIC_dict__ = {
    'left_folder' : 'Images_example/left_identification',
    'right_folder' : 'Images_example/right_identification',
    'name' : 'micro_identification',
    'saving_folder' : saving_folder,
    'window' : [[300, 1700], [300, 1700]]}
    
    # Create the list of z plans
    Folder = __calibration_dict__['left_folder']
    Imgs = sorted(glob(str(Folder) + '/*'))
    x3_list = np.zeros((len(Imgs)))
    for i in range (len(Imgs)) :
        x3_list[i] = float(Imgs[i][len(Folder)+ 1:-4])
    
    # Chose the degrees for Soloff and direct polynomial fitting
    Soloff_pform = 332
    direct_pform = 4
    
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
    direct_A, Mag = pcs.direct_calibration (__calibration_dict__,
                                            x3_list,
                                            direct_pform)
    
    print('')
    print('#####       ')
    print('Soloff method - Start calibration')
    print('#####       ')  
    A111, A_pol, Mag = pcs.Soloff_calibration (__calibration_dict__,
                                               x3_list,
                                               Soloff_pform)
    
    print('')
    print('#####       ')
    print('Identification of displacements field by DIC')
    print('#####       ')
    print('')
    Xleft_id, Xright_id = data.DIC_disflow(__DIC_dict__)
    Xleft_idc, Xright_idc = data.DIC_3D_composed_detection(__DIC_dict__)
    
    print('')
    print('#####       ')
    print('Calculation of 3D view')
    print('#####       ')
    print('')
    # Chose right and left coordinates
    X_c1 = Xleft_id[0]
    X_c2 = Xright_id[0]
    
    # Direct identification
    xDirect_solution = pcs.direct_identification (X_c1,
                                                  X_c2,
                                                  direct_A,
                                                  direct_pform)
    xD, yD, zD = xDirect_solution
    wnd = __DIC_dict__['window']    
    zD = zD.reshape((wnd[0][1] - wnd[0][0], wnd[1][1] - wnd[1][0]))
    
    plt.figure()
    plt.imshow(zD)
    plt.title('Z projected on left camera with direct calculation')
    cb = plt.colorbar()
    plt.clim(2.6, 3)
    cb.set_label('z in mm')
    plt.show() 
    
    
    # Soloff identification
    soloff_file = saving_folder + '/xsolution_soloff0.npy'
    
    # Condition not to calculate Soloff points if already exists in a
    # database
    if os.path.exists(soloff_file) :
        xSoloff_solution = np.load(soloff_file)
    else :
        xSoloff_solution = pcs.Soloff_identification (X_c1,
                                                      X_c2,
                                                      A111, 
                                                      A_pol,
                                                      Soloff_pform,
                                                      method = 'curve_fit')       
        np.save(soloff_file, xSoloff_solution)
    
    xS, yS, zS = xSoloff_solution
    zS = zS.reshape((wnd[0][1] - wnd[0][0], wnd[1][1] - wnd[1][0]))
      
    plt.figure()
    plt.imshow(zS)
    plt.title('Z projected on left camera with Soloff calculation')
    cb = plt.colorbar()
    plt.clim(2.6, 3)
    cb.set_label('z in mm')
    plt.show()
    
    AI_training_size = 50000
    model = pcs.AI_training (X_c1,
                             X_c2,
                             xSoloff_solution,
                             AI_training_size = AI_training_size)
    
    xAI_solution = pcs.AI_identification (X_c1,
                                          X_c2,
                                          model)

    xAI, yAI, zAI = xSoloff_solution
    zAI = zAI.reshape((wnd[0][1] - wnd[0][0], wnd[1][1] - wnd[1][0]))
      
    plt.figure()
    plt.imshow(zAI)
    plt.title('Z projected on left camera with AI calculation')
    cb = plt.colorbar()
    plt.clim(2.6, 3)
    cb.set_label('z in mm')
    plt.show()
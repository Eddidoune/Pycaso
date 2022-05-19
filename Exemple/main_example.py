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

import main
import data_library as data
import matplotlib.pyplot as plt

if __name__ == '__main__' :    
    # Define the inputs
    __calibration_dict__ = {
    'left_folder' : 'Images_example/left_calibration',
    'right_folder' : 'Images_example/right_calibration',
    'name' : 'micro_calibration',
    'ncx' : 16,
    'ncy' : 12,
    'sqr' : 0.3}
    
    __DIC_dict__ = {
    'left_folder' : 'Images_example/left_coin_identification',
    'right_folder' : 'Images_example/right_coin_identification',
    'name' : 'micro_identification',
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
    saving_folder = 'results/main_expl_300_1700'
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
    direct_A, Mag = main.direct_calibration (__calibration_dict__,
                                             x3_list,
                                             saving_folder,
                                             direct_pform,
                                             detection = True)
    
    print('')
    print('#####       ')
    print('Soloff method - Start calibration')
    print('#####       ')  
    A111, A_pol, Mag = main.Soloff_calibration (__calibration_dict__,
                                                x3_list,
                                                saving_folder,
                                                Soloff_pform = Soloff_pform,
                                                detection = False)
    
    print('')
    print('#####       ')
    print('Identification of displacements field by DIC')
    print('#####       ')
    print('')
    Xleft_id, Xright_id = data.DIC_3D_detection_lagrangian(__DIC_dict__, 
                                                           detection = True,
                                                           saving_folder = saving_folder)
        
    print('')
    print('#####       ')
    print('Calculation of 3D view')
    print('#####       ')
    print('')
    # Chose right and left coordinates
    X_c1 = Xleft_id[0]
    X_c2 = Xright_id[0]
    
    # Direct identification
    xDirect_solution = main.direct_identification (X_c1,
                                                   X_c2,
                                                   direct_A,
                                                   direct_pform = direct_pform)
    xD, yD, zD = xDirect_solution
    wnd = __DIC_dict__['window']    
    zD = zD.reshape((wnd[0][1] - wnd[0][0], wnd[1][1] - wnd[1][0]))
     
    plt.figure()
    plt.imshow(zD)
    plt.title('Z projected on left camera with direct calculation')
    cb = plt.colorbar()
    cb.set_label('z in mm')
    plt.show() 
    
    
    
    
    
    # Soloff identification
    soloff_file = saving_folder + '/xsolution_soloff0.npy'
    
    # Condition not to calculate Soloff points if already exists in a
    # database
    if os.path.exists(soloff_file) :
        xSoloff_solution = np.load(soloff_file)
    else :
        xSoloff_solution = main.Soloff_identification (X_c1,
                                                       X_c2,
                                                       A111, 
                                                       A_pol,
                                                       Soloff_pform = Soloff_pform,
                                                       method = 'curve_fit')       
        np.save(soloff_file, xSoloff_solution)
    
    xS, yS, zS = xSoloff_solution
    zS = zS.reshape((wnd[0][1] - wnd[0][0], wnd[1][1] - wnd[1][0]))
      
    plt.figure()
    plt.imshow(zS)
    plt.title('Z projected on left camera with Soloff calculation')
    cb = plt.colorbar()
    cb.set_label('z in mm')
    plt.show()  
    
    
    
    
    
    
    # AI identification
    # Chose the Number of Datas for Artificial Intelligence Learning
    AI_training_size = 50000
    # Create the .csv to make an AI identification
    model_file = saving_folder +'/Soloff_AI_0_' + str(AI_training_size) + '_points.csv'      
    AI_file = saving_folder + '/xsolution_AI0.npy'
    if os.path.exists(AI_file) :
        xAI_solution = np.load(AI_file)
    else :
        # Train the AI with already known points
        model = main.AI_training (X_c1,
                                  X_c2,
                                  xSoloff_solution,
                                  AI_training_size = AI_training_size,
                                  file = model_file)
        # Use the AI model to solve every points
        xAI_solution = main.AI_identification (X_c1,
                                               X_c2,
                                               model)
        np.save(AI_file, xAI_solution)
    
    xAI, yAI, zAI = xAI_solution
    zAI = zAI.reshape((wnd[0][1] - wnd[0][0], wnd[1][1] - wnd[1][0]))
    
    # Plot figure
    plt.figure()
    plt.imshow(zAI)
    plt.title('Z projected on left camera with AI calculation')
    cb = plt.colorbar()
    cb.set_label('z in mm')
    plt.show()  
             
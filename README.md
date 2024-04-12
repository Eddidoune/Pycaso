# Pycaso
PYthon module for CAlibration of cameras by SOloff’s method

Pycaso is a python code used to calibrate any pair of cameras using the method of Soloff (or the direct method). This method links detected points X from cameras 1 and 2 to the real position x in the 3D-space. It is divided in two parts :
- Calibration : A lot of coordinates (**X**=(X,Y) and **x**(x,y,z)) are known and a polynomial form P can be estimate for each coordinate X_i = P(**x**)
- Identification : With a Levenberg-Marcquardt algorithm, it is possible to build new points into the 3D-space using the detected points **X** and the polynomials P (**x** = f(P1(X1), P2(Y1), P3(X2), P4(Y2)) 


The file Exemple/main_example.py propose a typical protocol of resolution using few Pycaso functions.

# Requirements
To install Pycaso, you will need [Python 3](https://www.python.org/downloads/) (3.6 or higher) with the following modules :
- [numpy](https://numpy.org/install/)
- [Open CV](https://pypi.org/project/opencv-python/) for image acquisition and identification, 
- [matplotlib](https://matplotlib.org/) for graph plotting and scipy for Levenberg  Marcquardt  functions. 
- Others modules as [glob](https://docs.python.org/3/library/glob.html) , [deepcopy](https://docs.python.org/3/library/copy.html), [os](https://docs.python.org/3/library/os.html), [sigfig](https://pypi.org/project/sigfig/), [scikit-image](https://scikit-image.org/docs/dev/install.html) and [sys](https://docs.python.org/3/library/sys.html). 

# Installation
All the modules can be find on Github

# Illustrative example - Coin identification by Digital Image Correlation (DIC)
To illustrate the software, let’s try to identify the shape of a coin speckled with
a chalk solution.

## Generate pattern
The file pattern.py create a ChArUco chessboard which can be print for stereo
correlation. It takes as Input a dictionary with :
- ncx = 16 : The number of x squares for the chessboard
- ncy = 12 : The number of y squares for the chessboard
- pixel_factor = 10 : The number of sub-pixel per Charucco pixel
This file generate a png pattern which can be print on a plane surface for the calibration step.

## Device and acquisition
The inputs of the device are :
- The printed pattern (ChArUco chessboard)
- Two cameras
- A light source
- A z-axe control
- An acquisition support (Computer)
- A coin to identify

And the outputs are :
- 1) cam1 and cam2 pictures of the pattern during calibration stage. One pair of
pictures for each z coordinate.
- 2) cam1 and cam2 pictures of the coin during identification stage.

## Calibration of the volume of interest (VOI)
Use the cameras and the z-axe to take a lot of picture of the pattern at different position z (100 position here).
Save all of the cam1 image (resp cam2) in a folder 'Folder_calibration_cam1' with the z position as a name 'z.png'.
- Define the calibration dictionnary :
```
calibration_dict = {
								'cam1_folder' : 'Images_example/left_calibration11',
								'cam2_folder' : 'Images_example/right_calibration11',
								'name' : 'micro_calibration',								
								'saving_folder' : 'results/main_exemple',
								'ncx' : 16,
								'ncy' : 12,
								'pixel_factor' : 10}
```
 - Create the list of z plans
```
Folder = calibration_dict['cam1_folder']
Imgs = sorted(glob(str(Folder) + '/*'))
z_list = np.zeros((len(Imgs)))
for i in range (len(Imgs)) :
    z_list[i] = float(Imgs[i][len(Folder)+ 1:-4])
```

- Chose the degrees for Soloff and direct polynomial fitting
```
Soloff_pform = 332
direct_polynomial_form = 4
```
Create the result folder if not exist
```
saving_folder = 'results/main_exemple'
```
Lauch the Soloff calibration function in the main.py :
```
Soloff_constants0, Soloff_constants, Mag= Pycaso.Soloff_calibration (z_list,
																				Soloff_pform,
																				**calibration_dict)
```
And/Or the direct calibration function in the main.py :
```
direct_constants, Mag= Pycaso.direct_calibration (z_list,
																		direct_pform,
direct_constants, Mag= Pycaso.direct_calibration (z_list,																		**calibration_dict)
```
The calibration parameters are identified and calibration part is done. For more information about the resolution, see the Hessian detection explaination.

## Identification of the coin
Use the cameras to take some pair of pictures of the coin.
Save all of the cam1 image (resp cam2) in a folder 'Folder_identification_cam1'. Then, define a DIC dictionnary :
```
DIC_dict = {
					'cam1_folder' : 'Images_example/left_identification',
					'cam2_folder' : 'Images_example/right_identification',
					'name' : 'micro_identification',
					'saving_folder' : saving_folder,
					'window' : [[300, 1700], [300, 1700]]}
```

The identification can start :
First, use the correlation process (default = GCpu_OpticalFlow or disflow) from cam1 to cam2 images to identify DIC fields. With those fields, it is possible to detect a same point (pixel) on the cam1 and the cam2.
```
Xcam1_id, Xcam2_id = data.DIC_get_positions(DIC_dict)
```
Then use one of the pairs (Xcam1_id[0], Xcam2_id[0]) to create the points on the global referential (x,y,z) (Here show the Soloff method but the direct or Zernike method can be use) :
```
xSoloff_solution = Soloff_identification(Xcam1_id[0],
																Xcam2_id[0],
																Soloff_constants0, 
																Soloff_constants,
																Soloff_pform,
																method = 'curve_fit')       
xS, yS, zS = xSoloff_solution
```
Now all of the spacial points i are detected (xS[i], yS[i], zS[i]). 
Then it is possible to project the zS on the cam1 to tchek the cinematic field :
```
zS = np.reshape(zS,(1400,1400))
plt.imshow(zS)
```

## Identification by AI
In order to accelerate the resolution, it is possible to train AI on Soloff's evaluated points.
Let's chose for example 50 000 points for training the AI model :
NB : Here all the points are calculated by Soloff but, if you want to use the AI protocole to increase the speed of calculation, it is useless to use Soloff on all the points (only 50 000 is enough).
```
AI_training_size = 50000
model = AI_training (X_c1,
								   X_c2,
								   xSoloff_solution,
								   I_training_size = AI_training_size)
```

Then, all the points can be calculated here with the AI model :
```
xAI_solution = AI_identification (X_c1,
													   X_c2,
													   model)
xAI, yAI, zAI = xAI_solution
```
Now all of the spacial points i are detected (xAI[i], yAI[i], zAI[i]). 
Then it is possible to project the zAI on the cam1 to tchek the cinematic field :
```
zAI = np.reshape(zAI,(1400,1400))
plt.imshow(zAI)
```

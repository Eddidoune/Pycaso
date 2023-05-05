# Pycaso
PYthon module for CAlibration of cameras by SOloff’s method

Pycaso is a python code used to calibrate any pair of cameras using the method of Soloff (or the direct method). This method links detected points X from cameras left an right to the real position x in the 3D-space. It is divided in two parts :
- Calibration : A lot of coordinates (**X**=(X,Y) and **x**(x,y,z)) are known and a polynomial form P can be estimate for each coordinate X_i = P(**x**)
- Identification : With a Levenberg-Marcquardt algorithm, it is possible to build new points into the 3D-space using the detected points **X** and the polynomials P (**x** = f(P1(X1), P2(Y1), P3(X2), P4(Y2)) 


The file main.py propose a typical protocol of resolution using few Pycaso functions (full_Soloff_calibration, full_Soloff_identification, DIC_3D_detection_lagrangian etc...).

# Requirements
To install Pycaso, you will need [Python 3](https://www.python.org/downloads/) (3.6 or higher) with the following modules :
- [numpy], [Open CV], [matplotlib], [sigfig], [pandas], [scikit], [scipy] and [seaborn]
- The library [GCpu_OpticalFlow](https://github.com/chabibchabib/GCpu_OpticalFlow/blob/master/Doc/README.rst) is recommended.
- [cupy] could increase the speed of processing and is recommended too.

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
- 1) Right and left pictures of the pattern during calibration stage. One pair of
pictures for each z coordinate.
- 2) Right and left pictures of the coin during identification stage.

## Calibration of the volume of interest (VOI)
Use the cameras and the z-axe to take a lot of picture of the pattern at different position z (100 position here).
Save all of the left image (resp right) in a folder 'Folder_calibration_left' with the z position as a name 'z.png'.
- Define the calibration dictionnary :
```
calibration_dict = {
								'left_folder' : '/Folder_calibration_left',
								'right_folder' : '/Folder_calibration_right',
								'name' : 'Whatever',
								'ncx' : 16,
								'ncy' : 12,
								'pixel_factor' : 10}
```
 - Create the list of z plans
```
Folder = calibration_dict['left_folder']
Imgs = sorted(glob(str(Folder) + '/*'))
x3_list = np.zeros((len(Imgs)))
for i in range (len(Imgs)) :
    x3_list[i] = float(Imgs[i][len(Folder)+ 1:-4])
```

- Chose the degrees for Soloff and direct polynomial fitting
```
Soloff_pform = 332
direct_polynomial_form = 4
```
Create the result folder if not exist
```
saving_folder = '/saving_folder_name'
```
Lauch the Soloff calibration function in the main.py :
```
A111, A_pol, Mag= Pycaso.Soloff_calibration (x3_list,
																				Soloff_pform,
																				**calibration_dict)
```
And/Or the direct calibration function in the main.py :
```
direct_A, Mag= Pycaso.direct_calibration (x3_list,
																		direct_pform,
																		**calibration_dict)
```
The calibration parameters are identified and calibration part is done. For more information about the resolution, see the Hessian detection explaination.

## Identification of the coin
Use the cameras to take some pair of pictures of the coin.
Save all of the left image (resp right) in a folder 'Folder_identification_left'. Then, define a DIC dictionnary :
```
DIC_dict = {
					'left_folder' : '/Folder_identification_left',
					'right_folder' : '/Folder_identification_right',
					'name' : 'Whatever',
					'window' : [[300, 1700], [300, 1700]]}
```

The identification can start :
First, use the correlation process (default = GCpu_OpticalFlow or disflow) from left to right images to identify DIC fields. With those fields, it is possible to detect a same point (pixel) on the left and the right cameras.
```
Xleft_id, Xright_id = data.DIC_get_positions(DIC_dict)
```
Then use one of the pairs (Xleft_id[0], Xright_id[0]) to create the points on the global referential (x,y,z) :
```
xSoloff_solution = Soloff_identification(Xleft_id[0],
																Xright_id[0],
																A111, 
																A_pol,
																Soloff_pform,
																method = 'Peter')       
xS, yS, zS = xSoloff_solution
```
Now all of the spacial points i are detected (xS[i], yS[i], zS[i]). 
Then it is possible to project the zS on the left camera to tchek the cinematic field :
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

Then, all the points can be calculated here with the AI model but the Peter solver is still better than the AI method (faster and more accuracy) :
```
xAI_solution = AI_identification (X_c1,
													   X_c2,
													   model)
xAI, yAI, zAI = xAI_solution
```
Now all of the spacial points i are detected (xAI[i], yAI[i], zAI[i]). 
Then it is possible to project the zAI on the left camera to tchek the cinematic field :
```
zAI = np.reshape(zAI,(1400,1400))
plt.imshow(zAI)
```

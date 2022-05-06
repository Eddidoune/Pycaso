# Pycaso
PYthon module for CAlibration of cameras by SOloff’s method

Pycaso is a python code used to calibrate any pair of cameras using the method of Soloff (or the direct method). This method links detected points X from cameras left an right to the real position x in the 3D-space. It is divided in two parts :
- Calibration : A lot of coordinates (**X**=(X,Y) and **x**(x,y,z)) are known and a polynomial form P can be estimate for each coordinate X_i = P(**x**)
- Identification : With a Levenberg-Marcquardt algorithm, it is possible to build new points into the 3D-space using the detected points **X** and the polynomials P (**x** = f(P1(X1), P2(Y1), P3(X2), P4(Y2)) 


The file main.py propose a typical protocol of resolution using few Pycaso functions (full_Soloff_calibration, full_Soloff_identification, DIC_3D_detection_lagrangian etc...).

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
- ncx = 12 : The number of x squares for the chessboard
- ncy = 16 : The number of y squares for the chessboard
- dpi = 1200 : The dots per inch of the chessboard
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
__calibration_dict__ = {
'left_folder' : '/Folder_calibration_left',
'right_folder' : '/Folder_calibration_right',
'name' : 'Whatever',
'ncx' : 16,
'ncy' : 12,
 sqr' : 0.3}
 ```
 - Create the list of z plans
```
Folder = __calibration_dict__['left_folder']
Imgs = sorted(glob(str(Folder) + '/*'))
x3_list = np.zeros((len(Imgs)))
for i in range (len(Imgs)) :
    x3_list[i] = float(Imgs[i][len(Folder)+ 1:-4])
```

- Chose the degrees for Soloff and direct polynomial fitting
```
polynomial_form = 332
direct_polynomial_form = 4
```
Create the result folder if not exist
```
saving_folder = '/saving_folder_name'
```
Lauch the Soloff calibration function in the main.py :
```
A111, A_pol, Mag= Soloff_calibration (__calibration_dict__,
                          	       x3_list,
                 	  	       saving_folder,
                                      polynomial_form = polynomial_form,
                                      detection = False)
``` 
And/Or the direct calibration function in the main.py :
```
direct_A, Mag= direct_calibration (__calibration_dict__,
                        	    x3_list,
                                   saving_folder,
                                   direct_polynomial_form,
                                   detection = False)
``` 
The calibration parameters are identified and calibration part is done. For more information about the resolution, see the Hessian detection explaination.

## Identification of the coin
Use the cameras to take some pair of pictures of the coin.
Save all of the left image (resp right) in a folder 'Folder_identification_left'. Then, define a DIC dictionnary :
```
__DIC_dict__ = {
'left_folder' : '/Folder_identification_left',
'right_folder' : '/Folder_identification_right',
'name' : 'Whatever',
'window' : [[300, 1700], [300, 1700]]}
```

The identification can start :
First, use the a correlation process (Here disflow) from left to right images to identify DIC fields. With those fields, it is possible to detect a same point (pixel) on the left and the right cameras.
```
Xleft_id, Xright_id = data.DIC_3D_detection(__DIC_dict__, 
	                                    detection = True,
	                                    saving_folder = saving_folder)
```
Then use one of the pairs (0) to create the points on the global referential (x,y,z) :
```
xSoloff_solution = Soloff_identification(Xleft_id[0],
                                         Xright_id[0],
                                         A111, 
                                         A_pol,
                                         polynomial_form = polynomial_form,
                                         method = 'curve_fit')       
xS, yS, zS = xSoloff_solution
```
Now all of the spacial points i are detected (xS[i], yS[i], zS[i]). 
Then it is possible to project the zS on the left camera to tchek the cinematic field :
```
zS = np.reshape(zS,(1400,1400))
plt.imshow(zS)
```



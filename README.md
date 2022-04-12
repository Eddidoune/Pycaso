# Pycaso
PYthon module for CAlibration of cameras by SOloff’s method

Pycaso is a python code used to calibrate any pair of cameras using the method of Soloff (or the direct method). This method links detected points X from cameras left an right to the real position x in the 3D-space. It is divided in two parts :
- Calibration : A lot of coordinates (**X**=(X,Y) and **x**(x,y,z)) are known and a polynomial form P can be estimate for each coordinate X_i = P(**x**)
- Identification : With a Levenberg-Marcquardt algorithm, it is possible to build new points into the 3D-space using the detected points **X** and the polynomials P (**x** = f(P1(X1), P2(Y1), P3(X2), P4(Y2))


The file main.py propose a typical protocol of resolution using few Pycaso functions (full_Soloff_calibration, full_Soloff_identification, DIC_3D_detection_lagrangian etc...).

# Methodologie

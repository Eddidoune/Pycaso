# Pattern detection
The ChAruco's pattern detection is first automatically resolved with the OpenCV library in the file data_library.py with the function Calibrate(__dict__).calibrate(self, image). For more information, check the [OpenCV docs](https://docs.opencv.org/3.4/d9/df8/tutorial_root.html).
![Automatic detection](https://github.com/Eddidoune/Pycaso/blob/main/src/Pattern%20detection%20explaination/Hessian1.png)
For all the points not detected by this method (because of default on the pattern or bad picture catching), another detection need to be add.


# Hessian holes detection

## 1. Filter the image
Use the Hessian determinant as filter parameter (eigenvalues product)
![Hessian determinant filter](https://github.com/Eddidoune/Pycaso/blob/main/src/Pattern%20detection%20explaination/Hessian0.png)

## 2. Find the referential
- Chose two points A(Magenta) and B(Yellow) which are not on the same line or column.
![](https://github.com/Eddidoune/Pycaso/blob/main/src/Pattern%20detection%20explaination/Hessian2.png)

- Make the nx and ny vectors
![](https://github.com/Eddidoune/Pycaso/blob/main/src/Pattern%20detection%20explaination/Hessian3.png)

- Find the origin O (Green) from A
![](https://github.com/Eddidoune/Pycaso/blob/main/src/Pattern%20detection%20explaination/Hessian4.png)


## 3. Find the missing points
- Look around the likely position of a missing point. Then look at the local hessian filter and find the baricenter of the biggest shape
![](https://github.com/Eddidoune/Pycaso/blob/main/src/Pattern%20detection%20explaination/Hessian5.png.png)

- Iterate and define all the missing points (Blue)
![](https://github.com/Eddidoune/Pycaso/blob/main/src/Pattern%20detection%20explaination/Hessian6.png)


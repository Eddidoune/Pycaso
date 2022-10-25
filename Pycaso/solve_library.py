#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sigfig as sgf
import pandas as pd
import os

try : 
    import cupy as np
    cpy = True
except ImportError:
    import numpy as np
    cpy = False
    
import matplotlib.pyplot as plt
import scipy.optimize as sopt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

  
class Direct_Polynome(dict) :
    def __init__(self, _dict_):
        self._dict_ = _dict_
        self.polynomial_form = _dict_['polynomial_form']

    def pol_form (self, Xl, Xr) :
        """Create the matrix M = f(Xl,Xr) with f the polynomial function of 
        degree n
        
        Args:
           Xl : numpy.ndarray
               Left detected points  Xl(Xl1, Xl2)
           Xr : numpy.ndarray
               Right detected points  Xr(Xr1, Xr2)
               
        Returns:
           M : numpy.ndarray
               M = f(Xl,Xr)
        """
        polynomial_form = self.polynomial_form
        Xl1, Xl2 = Xl
        Xr1, Xr2 = Xr
        
        n = len(Xl1)
        if   polynomial_form == 1 :
            M = np.asarray ([np.ones((n)),   Xl1,           Xl2,            Xr1,            Xr2])

        elif polynomial_form == 2 :
            Xl12 = Xl1 * Xl1
            Xl22 = Xl2 * Xl2
            Xr12 = Xr1 * Xr1
            Xr22 = Xr2 * Xr2
            M = np.asarray ([np.ones((n)),   Xl1,           Xl2,            Xr1,            Xr2,
                             Xl12,           Xl1*Xl2,       Xl1*Xr1,        Xl1*Xr2,        Xl22,
                             Xl2*Xr1,        Xl2*Xr2,       Xr12,           Xr1*Xr2,        Xr22])

        elif polynomial_form == 3 :
            Xl12 = Xl1 * Xl1
            Xl13 = Xl1 * Xl1 * Xl1
            Xl22 = Xl2 * Xl2
            Xl23 = Xl2 * Xl2 * Xl2
            Xr12 = Xr1 * Xr1
            Xr13 = Xr1 * Xr1 * Xr1
            Xr22 = Xr2 * Xr2
            Xr23 = Xr2 * Xr2 * Xr2
            M = np.asarray ([np.ones((n)),   Xl1,           Xl2,            Xr1,            Xr2,
                             Xl12,           Xl1*Xl2,       Xl1*Xr1,        Xl1*Xr2,        Xl22,
                             Xl2*Xr1,        Xl2*Xr2,       Xr12,           Xr1*Xr2,        Xr22,
                             Xl13,           Xl12*Xl2,      Xl12*Xr1,       Xl12*Xr2,       Xl1*Xl22,
                             Xl1*Xl2*Xr1,    Xl1*Xl2*Xr2,   Xl1*Xr12,       Xl1*Xr1*Xr2,    Xl1*Xr22,
                             Xl23,           Xl22*Xr1,      Xl22*Xr2,       Xl2*Xr12,       Xl2*Xr1*Xr2,    
                             Xl2*Xr22,       Xr13,          Xr12*Xr2,       Xr1*Xr22,       Xr23])

        elif polynomial_form == 4 :
            Xl12 = Xl1 * Xl1
            Xl13 = Xl1 * Xl1 * Xl1
            Xl14 = Xl1 * Xl1 * Xl1 * Xl1
            Xl22 = Xl2 * Xl2
            Xl23 = Xl2 * Xl2 * Xl2
            Xl24 = Xl2 * Xl2 * Xl2 * Xl2
            Xr12 = Xr1 * Xr1
            Xr13 = Xr1 * Xr1 * Xr1
            Xr14 = Xr1 * Xr1 * Xr1 * Xr1
            Xr22 = Xr2 * Xr2
            Xr23 = Xr2 * Xr2 * Xr2
            Xr24 = Xr2 * Xr2 * Xr2 * Xr2
            M = np.asarray ([np.ones((n)),   Xl1,           Xl2,            Xr1,            Xr2,
                             Xl12,           Xl1*Xl2,       Xl1*Xr1,        Xl1*Xr2,        Xl22,
                             Xl2*Xr1,        Xl2*Xr2,       Xr12,           Xr1*Xr2,        Xr22,
                             Xl13,           Xl12*Xl2,      Xl12*Xr1,       Xl12*Xr2,       Xl1*Xl22,
                             Xl1*Xl2*Xr1,    Xl1*Xl2*Xr2,   Xl1*Xr12,       Xl1*Xr1*Xr2,    Xl1*Xr22,
                             Xl23,           Xl22*Xr1,      Xl22*Xr2,       Xl2*Xr12,       Xl2*Xr1*Xr2,    
                             Xl2*Xr22,       Xr13,          Xr12*Xr2,       Xr1*Xr22,       Xr23,
                             Xl14,           Xl13*Xl2,      Xl13*Xr1,       Xl13*Xr2,       Xl12*Xl22,
                             Xl12*Xl2*Xr1,   Xl12*Xl2*Xr2,  Xl12*Xr12,      Xl12*Xr1*Xr2,   Xl12*Xr22,
                             Xl1*Xl23,       Xl1*Xl22*Xr1,  Xl1*Xl22*Xr2,   Xl1*Xl2*Xr12,   Xl1*Xl2*Xr1*Xr2,
                             Xl1*Xl2*Xr22,   Xl1*Xr13,      Xl1*Xr12*Xr2,   Xl1*Xr1*Xr22,   Xl1*Xr23,
                             Xl24,           Xl23*Xr1,      Xl23*Xr2,       Xl22*Xr12,      Xl22*Xr1*Xr2,
                             Xl22*Xr22,      Xl2*Xr13,      Xl2*Xr12*Xr2,   Xl2*Xr1*Xr22,   Xl2*Xr23,
                             Xr14,           Xr13*Xr2,      Xr12*Xr22,      Xr1*Xr23,       Xr24])

        elif polynomial_form == 5 :
            Xl12 = Xl1 * Xl1
            Xl13 = Xl1 * Xl1 * Xl1
            Xl14 = Xl1 * Xl1 * Xl1 * Xl1
            Xl15 = Xl1 * Xl1 * Xl1 * Xl1 * Xl1
            Xl22 = Xl2 * Xl2
            Xl23 = Xl2 * Xl2 * Xl2
            Xl24 = Xl2 * Xl2 * Xl2 * Xl2
            Xl25 = Xl2 * Xl2 * Xl2 * Xl2 * Xl2
            Xr12 = Xr1 * Xr1
            Xr13 = Xr1 * Xr1 * Xr1
            Xr14 = Xr1 * Xr1 * Xr1 * Xr1
            Xr15 = Xr1 * Xr1 * Xr1 * Xr1 * Xr1
            Xr22 = Xr2 * Xr2
            Xr23 = Xr2 * Xr2 * Xr2
            Xr24 = Xr2 * Xr2 * Xr2 * Xr2
            Xr25 = Xr2 * Xr2 * Xr2 * Xr2 * Xr2
            M = np.asarray ([np.ones((n)),   Xl1,           Xl2,            Xr1,            Xr2,
                             Xl12,           Xl1*Xl2,       Xl1*Xr1,        Xl1*Xr2,        Xl22,
                             Xl2*Xr1,        Xl2*Xr2,       Xr12,           Xr1*Xr2,        Xr22,
                             Xl13,           Xl12*Xl2,      Xl12*Xr1,       Xl12*Xr2,       Xl1*Xl22,
                             Xl1*Xl2*Xr1,    Xl1*Xl2*Xr2,   Xl1*Xr12,       Xl1*Xr1*Xr2,    Xl1*Xr22,
                             Xl23,           Xl22*Xr1,      Xl22*Xr2,       Xl2*Xr12,       Xl2*Xr1*Xr2,    
                             Xl2*Xr22,       Xr13,          Xr12*Xr2,       Xr1*Xr22,       Xr23,
                             Xl14,           Xl13*Xl2,      Xl13*Xr1,       Xl13*Xr2,       Xl12*Xl22,
                             Xl12*Xl2*Xr1,   Xl12*Xl2*Xr2,  Xl12*Xr12,      Xl12*Xr1*Xr2,   Xl12*Xr22,
                             Xl1*Xl23,       Xl1*Xl22*Xr1,  Xl1*Xl22*Xr2,   Xl1*Xl2*Xr12,   Xl1*Xl2*Xr1*Xr2,
                             Xl1*Xl2*Xr22,   Xl1*Xr13,      Xl1*Xr12*Xr2,   Xl1*Xr1*Xr22,   Xl1*Xr23,
                             Xl24,           Xl23*Xr1,      Xl23*Xr2,       Xl22*Xr12,      Xl22*Xr1*Xr2,
                             Xl22*Xr22,      Xl2*Xr13,      Xl2*Xr12*Xr2,   Xl2*Xr1*Xr22,   Xl2*Xr23,
                             Xr14,           Xr13*Xr2,      Xr12*Xr22,      Xr1*Xr23,       Xr24,
                             Xl15,           Xl14*Xl2,      Xl14*Xr1,       Xl14*Xr2,       Xl13*Xl22,
                             Xl13*Xl2*Xr1,   Xl13*Xl2*Xr2,  Xl13*Xr12,      Xl13*Xr1*Xr2,   Xl13*Xr22,
                             Xl2*Xl23,       Xl2*Xl22*Xr1,  Xl2*Xl22*Xr2,   Xl2*Xl2*Xr12,   Xl2*Xl2*Xr1*Xr2,
                             Xl2*Xl2*Xr22,   Xl2*Xr13,      Xl2*Xr12*Xr2,   Xl2*Xr1*Xr22,   Xl2*Xr23,
                             Xl1*Xl24,       Xl1*Xl23*Xr1,  Xl1*Xl23*Xr2,   Xl1*Xl22*Xr12,  Xl1*Xl22*Xr1*Xr2,
                             Xl1*Xl22*Xr22,  Xl1*Xl2*Xr13,  Xl1*Xl2*Xr12*Xr2,Xl1*Xl2*Xr1*Xr22,Xl1*Xl2*Xr23,
                             Xl25,           Xl24*Xr1,      Xl24*Xr2,       Xl23*Xr12,      Xl23*Xr1*Xr2,
                             Xl23*Xr22,      Xl22*Xr13,     Xl22*Xr12*Xr2,  Xl22*Xr1*Xr22,  Xl22*Xr23,
                             Xl2*Xr14,       Xl2*Xr13*Xr2,  Xl2*Xr12*Xr22,  Xl2*Xr1*Xr23,   Xl2*Xr24,
                             Xr15,           Xr14*Xr2,      Xr13*Xr22,      Xr12*Xr23,      Xr1*Xr24,
                             Xr25])

        return (M)
    
   
class Soloff_Polynome(dict) :
    def __init__(self, _dict_) :
        self._dict_ = _dict_
        self.polynomial_form = _dict_['polynomial_form']

    def pol_form (self, x) :
        """Create the matrix M = f(x) with f the polynomial function of degree 
        (aab : a for x1, x2 and b for x3)
        
        Args:
           x : numpy.ndarray
               Real points x(x1, x2, x3)
           
        Returns:
           M : numpy.ndarray
               M = f(x)
        """
        polynomial_form = self.polynomial_form
        x1,x2,x3 = x
        n = len(x1)
        if   polynomial_form == 111 or polynomial_form == 1 :
            M = np.asarray ([np.ones((n)),   x1,        x2,        x3])
        elif polynomial_form == 221 :
            x12 = x1 * x1
            x22 = x2 * x2
            M = np.asarray ([np.ones((n)),   x1,        x2,        x3,         x12,
                             x1 *x2,         x22,       x1*x3,     x2*x3])   
            
        elif polynomial_form == 222 or polynomial_form == 2 :
            x12 = x1 * x1
            x22 = x2 * x2
            x32 = x3 * x3
            M = np.asarray ([np.ones((n)),   x1,        x2,        x3,         x1**2,
                             x1 *x2,         x2**2,     x1*x3,     x2*x3,      x32])  
            
        elif polynomial_form == 332 :
            x12 = x1 * x1
            x22 = x2 * x2
            x32 = x3 * x3
            x13 = x1 * x1 * x1
            x23 = x2 * x2 * x2
            M = np.asarray ([np.ones((n)),   x1,        x2,         x3,        x12,
                             x1 *x2,         x22,       x1*x3,      x2*x3,     x32,
                             x13,            x12*x2,    x1*x22,     x23,       x12*x3,
                             x1*x2*x3,       x22*x3,    x1*x32,     x2*x32])  
            
        elif polynomial_form == 333 or polynomial_form == 3 :
            x12 = x1 * x1
            x22 = x2 * x2
            x32 = x3 * x3
            x13 = x1 * x1 * x1
            x23 = x2 * x2 * x2
            x33 = x3 * x3 * x3
            M = np.asarray ([np.ones((n)),   x1,        x2,         x3,        x12,
                             x1 *x2,         x22,       x1*x3,      x2*x3,     x32,
                             x13,            x12*x2,    x1*x22,     x23,       x12*x3,
                             x1*x2*x3,       x22*x3,    x1*x32,     x2*x32,    x33]) 
            
        elif polynomial_form == 443 :
            x12 = x1 * x1
            x22 = x2 * x2
            x32 = x3 * x3
            x13 = x1 * x1 * x1
            x23 = x2 * x2 * x2
            x33 = x3 * x3 * x3
            x14 = x1 * x1 * x1 * x1
            x24 = x2 * x2 * x2 * x2
            M = np.asarray ([np.ones((n)),   x1,            x2,         x3,        x12,
                             x1 *x2,         x22,           x1*x3,      x2*x3,     x32,
                             x13,            x12*x2,        x1*x22,     x23,       x12*x3,
                             x1*x2*x3,       x22*x3,        x1*x32,     x2*x32,    x33,
                             x14,            x13*x2,        x12*x22,    x1*x23,    x24,
                             x13*x3,         x12*x2*x3,    x1*x22*x3,  x23*x3,    x12*x32,
                             x1*x2*x32,      x22*x32,       x1*x33,     x2*x33])  
            
        elif polynomial_form == 444 or polynomial_form == 4 :
            x12 = x1 * x1
            x22 = x2 * x2
            x32 = x3 * x3
            x13 = x1 * x1 * x1
            x23 = x2 * x2 * x2
            x33 = x3 * x3 * x3
            x14 = x1 * x1 * x1 * x1
            x24 = x2 * x2 * x2 * x2
            x34 = x3 * x3 * x3 * x3
            M = np.asarray ([np.ones((n)),   x1,            x2,         x3,        x12,
                             x1 *x2,         x22,           x1*x3,      x2*x3,     x32,
                             x13,            x12*x2,        x1*x22,     x23,       x12*x3,
                             x1*x2*x3,       x22*x3,        x1*x32,     x2*x32,    x33,
                             x14,            x13*x2,        x12*x22,    x1*x23,    x24,
                             x13*x3,         x12*x2*x3,    x1*x22*x3,  x23*x3,    x12*x32,
                             x1*x2*x32,      x22*x32,       x1*x33,     x2*x33,    x34])
            
        elif polynomial_form == 554 :
            x12 = x1 * x1
            x22 = x2 * x2
            x32 = x3 * x3
            x13 = x1 * x1 * x1
            x23 = x2 * x2 * x2
            x33 = x3 * x3 * x3
            x14 = x1 * x1 * x1 * x1
            x24 = x2 * x2 * x2 * x2
            x34 = x3 * x3 * x3 * x3
            x15 = x1 * x1 * x1 * x1 * x1
            x25 = x2 * x2 * x2 * x2 * x2
            M = np.asarray ([np.ones((n)),   x1,            x2,             x3,             x12,
                             x1 *x2,         x22,           x1*x3,          x2*x3,          x32,
                             x13,            x12*x2,        x1*x22,         x23,            x12*x3,
                             x1*x2*x3,       x22*x3,        x1*x32,         x2*x32,         x33,
                             x14,            x13*x2,        x12*x22,        x1*x23,         x24,
                             x13*x3,         x12*x2*x3,     x1*x22*x3,      x23*x3,         x12*x32,
                             x1*x2*x32,      x22*x32,       x1*x33,         x2*x33,         x34,
                             x15,            x14*x2,        x13*x22,        x12*x23,        x1*x24,
                             x25,            x14*x3,        x13*x2*x3,      x12*x22*x3,     x1*x23*x3, 
                             x24*x3,         x13*x32,       x12*x2*x32,     x1*x22*x32,     x24*x32,   
                             x12*x33,        x1*x2*x33,     x22*x33,        x1*x34,         x2*x34])  
            
        elif polynomial_form == 555 or polynomial_form == 5 :
            x12 = x1 * x1
            x22 = x2 * x2
            x32 = x3 * x3
            x13 = x1 * x1 * x1
            x23 = x2 * x2 * x2
            x33 = x3 * x3 * x3
            x14 = x1 * x1 * x1 * x1
            x24 = x2 * x2 * x2 * x2
            x34 = x3 * x3 * x3 * x3
            x15 = x1 * x1 * x1 * x1 * x1
            x25 = x2 * x2 * x2 * x2 * x2
            x35 = x3 * x3 * x3 * x3 * x3
            M = np.asarray ([np.ones((n)),   x1,            x2,             x3,             x12,
                             x1 *x2,         x22,           x1*x3,          x2*x3,          x32,
                             x13,            x12*x2,        x1*x22,         x23,            x12*x3,
                             x1*x2*x3,       x22*x3,        x1*x32,         x2*x32,         x33,
                             x14,            x13*x2,        x12*x22,        x1*x23,         x24,
                             x13*x3,         x12*x2*x3,     x1*x22*x3,      x23*x3,         x12*x32,
                             x1*x2*x32,      x22*x32,       x1*x33,         x2*x33,         x34,
                             x15,            x14*x2,        x13*x22,        x12*x23,        x1*x24,
                             x25,            x14*x3,        x13*x2*x3,      x12*x22*x3,     x1*x23*x3, 
                             x24*x3,         x13*x32,       x12*x2*x32,     x1*x22*x32,     x24*x32,   
                             x12*x33,        x1*x2*x33,     x22*x33,        x1*x34,         x2*x34,
                             x35])
            
        return (M)


    def polynomial_LM_CF (self, a, *x) :
        """Definition of the functionnal F (for curve_fit method)
        
        Args:
           x : numpy.ndarray
               Real points x(x1, x2, x3)
           a : numpy.ndarray
               cst of the polynomial function M = f(x)
           
        Returns:
           Xc : numpy.ndarray
               Calculted position
        """
        polynomial_form = self.polynomial_form
        x = np.array ([x])
        x = x.reshape((3,len(x[0])//3))
        M = Soloff_Polynome({'polynomial_form' : polynomial_form}).pol_form(x)    
        Xc = np.matmul(a, M)
        Xc = Xc.reshape(4*len(x[0]))
        return (Xc)
    
    def polynomial_LM_LS (self, x, X, a) :
        """Definition of the functionnal F (for least_squares method)
        
        Args:
           x : numpy.ndarray
               Real points x(x1, x2, x3)
           X : numpy.ndarray
               Measured points X(Xl1, Xl2, Xr1, Xr2)
           a : numpy.ndarray
               cst of the polynomial function M = f(x)
           
        Returns:
           X-Xc : numpy.ndarray
               Functional calculation
        """
        polynomial_form = self.polynomial_form
        x = np.array ([x])
        x = x.reshape((3,len(x[0])//3))
        M = Soloff_Polynome({'polynomial_form' : polynomial_form}).pol_form(x) 
        Xc = np.matmul(a, M)
        Xc = Xc.reshape(4*len(x[0]))
        F = X-Xc
        return (F)
    
    def polynomial_system (self, x, a) :
        """Create the matrix M = f(x) with f the polynomial function of degree 
        (aab : a for x1, x2 and b for x3)
        
        Args:
           x : numpy.ndarray
               Real points x(x1, x2, x3)
           a : numpy.ndarray
               cst of the polynomial function M = f(x)
           
        Returns:
           M : numpy.ndarray
               M = f(x)
        """
        polynomial_form = self.polynomial_form
        M = Soloff_Polynome({'polynomial_form' : polynomial_form}).pol_form(x)   
        X = np.matmul(a, M)
            
        return(X)    


def fit_plan_to_points(point,
                       title = False,
                       plotting = False) :
    """Plot the median plan from a serie of points
    
    Args:
       point : numpy.ndarray (shape = m,3)
           Real points x(x1, x2, x3)       
       title : str
           Title of the plotted figure
       plotting = Bool
           Plot the result or not
            
    Returns:
       plot points + associated plan
    """
    xs, ys, zs = point 
    
    try : 
        import cupy as np
        xsnp = np.asnumpy(xs)
        ysnp = np.asnumpy(ys)
        zsnp = np.asnumpy(zs)
    except ImportError:
        import numpy as np
        xsnp = xs
        ysnp = ys
        zsnp = zs
    
    # do fit
    tmp_A = []
    tmp_b = []
    for i in range(len(xs)):
        tmp_A.append([xs[i], ys[i], 1])
        tmp_b.append(zs[i])
    b = np.matrix(tmp_b).T
    A = np.matrix(tmp_A)
    
    # Manual solution
    fit = (A.T * A).I * A.T * b
    errors = b - A * fit
    residual = np.linalg.norm(errors)
    mean_error = np.mean (abs(errors))
    errors = np.reshape(errors, (len(errors)))
    
    # plot plan
    X,Y = np.meshgrid(np.linspace(np.min(xs), np.max(xs), 10),
                      np.linspace(np.min(ys), np.max(ys), 10))
    Z = np.zeros(X.shape)
    for r in range(X.shape[0]):
        for c in range(X.shape[1]):
            Z[r,c] = fit[0] * X[r,c] + fit[1] * Y[r,c] + fit[2]
    
    if plotting :
        from mpl_toolkits.mplot3d import Axes3D #<-- Note the capitalization! 
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(xsnp, ysnp, zsnp, color='b')
        ax.plot_wireframe(X,Y,Z, color='k')
        if title :
            ax.set_title(title)
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_zlabel('z (mm)')
    
    fit = np.transpose(np.array(fit))[0]
    
    return (fit, errors, mean_error, residual)

def fit_plans_to_points(points, 
                        title = False,
                        plotting = False):
    """Plot the medians plans from series of points
    
    Args:
       points : numpy.ndarray (shape = l,m,3)
           Real points x(x1, x2, x3)       
       title : str
           Title of the plotted figures
       plotting = Bool
           Plot the result or not
            
    Returns:
       plot points + associated plans
    """
    # plot raw data
    l = len (points)
    fit = np.zeros((l, 3))
    errors = []
    mean_error = np.zeros(l)
    residual = np.zeros(l)
    maxerror = []
    for i in range (len(points)) :
        point = points[i]
        fit[i], errori, mean_error[i], residual[i] = fit_plan_to_points(point, 
                                                                        title = title,
                                                                        plotting = plotting)
        maxerror.append(np.max(abs(errori)))
        errors.append(errori)
    if plotting :
        plt.figure()
        plt.show()    
    print('Plan square max error = ', sgf.round((max(maxerror)), sigfigs =3), ' mm')
    print('Plan square mean error = ', sgf.round((np.mean(mean_error**2))**(1/2), sigfigs = 3), ' mm')
    print('Plan square mean residual = ', sgf.round((np.mean(residual**2))**(1/2), sigfigs = 3))
    
    return (fit, errors, mean_error, residual)

def refplans(xc1, 
             x3_list,
             plotting = False) :
    """Plot the medians plans from references points
    
    Args:
       xc1 : numpy.ndarray (shape = 3,n)
           Real points x(x1, x2, x3)       
       x3_list : numpy.ndarray
           List of the different plans coordinates
       plotting = Bool
           Plot the result or not
            
    Returns:
       plot points + associated plans
    """
    m, n = xc1.shape
    x,y,z = xc1
    xcons = []
    p0, pf = 0, 0
    for z_i in x3_list :
        while z[pf] == z_i :
            pf += 1
            if pf > n-1 :
                break
        plan = np.array ([x[p0:pf], y[p0:pf], z[p0:pf]])
        p0 = pf
        xcons.append (plan)
    fit_plans_to_points(xcons, 
                        title = 'Calibration plans',
                        plotting = plotting)

def least_square_method (Xc1_identified, 
                         Xc2_identified, 
                         A111) :
    """Resolve by least square method the system A . x = X for each points 
    detected and both cameras
    
    Args:
       Xc1_identified : numpy.ndarray
           Real positions of camera 1
       Xc2_identified : numpy.ndarray
           Real positions of camera 2
       A111 : numpy.ndarray
           Constants of the first order calibration polynome
           
    Returns:
       x0 : numpy.ndarray
           Solution x = xsol of the system 
    """
    N = len (Xc1_identified)
    x0 = np.zeros((3, N))
    a1c1, a2c1 = A111[0,0,:], A111[0,1,:]
    a1c2, a2c2 = A111[1,0,:], A111[1,1,:]
    A = np.array([a1c1, a2c1, a1c2, a2c2])
    A = A[:,1:4]
    At = np.transpose (A)  
    J = np.matmul(At, A)
    J_ = np.linalg.inv(J)
    
    for i in range (N) :
        X1c1, X2c1 = Xc1_identified[i,0], Xc1_identified[i,1]
        X1c2, X2c2 = Xc2_identified[i,0], Xc2_identified[i,1]
        X = np.array([X1c1-a1c1[0], X2c1-a2c1[0], X1c2-a1c2[0], X2c2-a2c2[0]])
        XA = np.matmul(X, A)
        x0i = np.matmul(J_, XA)
        x0[:, i] = x0i
    
    return (x0)    

def xopt_mlib (Xtuple) :
    """Multiprocessing function used on the next function Levenberg_Marquardt_solving.
    
    Args:
        Xtuple : list
            List of arguments for multiprocessing
           
    Returns:
        xopt : numpy.ndarray
            Solution of the LM resolution
    """
    Xdetected, x0_part, Soloff_pform, A0 = Xtuple
    Ns = Xdetected.shape[1]
    xopt = np.zeros((3*Ns))
    Xdetected_part = Xdetected
    for i in range (Xdetected_part.shape[1]) :
        X0i = Xdetected_part[:,i]
        x0i = x0_part[:,i]
        xopti, pcov = sopt.curve_fit(Soloff_Polynome({'polynomial_form' : Soloff_pform}).polynomial_LM_CF, 
                                    A0, 
                                    X0i, 
                                    p0 = x0i, 
                                    method ='lm')
        xopt[i], xopt[Ns + i], xopt[2*Ns + i] = xopti
    return (xopt)

def Levenberg_Marquardt_solving (Xc1_identified, 
                                 Xc2_identified, 
                                 A, 
                                 x0, 
                                 Soloff_pform, 
                                 method = 'curve_fit') :
    """Resolve by Levenberg-Marcquardt method the system A . x = X for each 
    points detected and both cameras
    
    Args:
        Xc1_identified : numpy.ndarray
            Real positions of camera 1
        Xc2_identified : numpy.ndarray
            Real positions of camera 2
        A : numpy.ndarray
            Constants of the calibration polynome
        x0 : numpy.ndarray
            Initial guess
        Soloff_pform : int
            Polynomial form
        method : str
            Chosen method of resolution. Can take 'curve_fit' or 'least_squares'
           
    Returns:
        xopt : numpy.ndarray
            Solution of the LM resolution
        Xcalculated : numpy.ndarray
            Solution calculated
        Xdetected : numpy.ndarray
            Solution detected (Xc1_identified, Xc2_identified)
    """   
    try : 
        from multiprocessing import Pool
        mlib = True
    except ImportError:
        mlib = False 
        try : 
            from joblib import Parallel, delayed
            jlib = True
        except ImportError:
            jlib = False    
    
        
    core_number = os.cpu_count()

    N = len(x0[0])    
    Xdetected = np.array([Xc1_identified[:,0], 
                          Xc1_identified[:,1], 
                          Xc2_identified[:,0], 
                          Xc2_identified[:,1]])
    A0 = np.array([A[0,0], A[0,1], A[1,0], A[1,1]])
    xopt = np.zeros((3,N))
    
    win_size = Xdetected.shape[1]/core_number
    slices = []
    for i in range (core_number) :
        start = i*win_size
        if i == core_number-1 :
            slices.append(slice(int(round(start)), Xdetected.shape[1]))
        else :            
            slices.append(slice(int(round(start)), int(round(start + win_size))))
    
    if mlib :       
        with Pool(core_number) as p :
            xtuple = []
            for i in range (core_number) :
                sl = slices[i]
                Xti = Xdetected[:, sl]
                x0i = x0[:,sl]
                xtuple.append((Xti, x0i,Soloff_pform, A0))
            xopt_parallel = p.map(xopt_mlib, xtuple)
            
        for part in range (len(xopt_parallel)) :
            sl = slices[part]
            xopt_part = xopt_parallel[part]
            xopt[:,sl] = xopt_part.reshape((3,sl.stop - sl.start))

    elif jlib :
        def xopt_solve (Xdetected, sl) :
            Ns = sl.stop - sl.start
            xopt = np.zeros((3*Ns))
            Xdetected_part = Xdetected[:,sl]
            x0_part = x0[:,sl]
            for i in range (Xdetected_part.shape[1]) :
                X0i = Xdetected_part[:,i]
                x0i = x0_part[:,i]
                xopti, pcov = sopt.curve_fit(Soloff_Polynome({'polynomial_form' : Soloff_pform}).polynomial_LM_CF, 
                                            A0, 
                                            X0i, 
                                            p0 = x0i, 
                                            method ='lm')
                xopt[i], xopt[Ns + i], xopt[2*Ns + i] = xopti
            return (xopt)
        xopt_parallel = Parallel(n_jobs=8)(delayed(xopt_solve)(Xdetected, sl) for sl in slices)
        
        for part in range (len(xopt_parallel)) :
            sl = slices[part]
            xopt_part = xopt_parallel[part]
            xopt[:,sl] = xopt_part.reshape((3,sl.stop - sl.start))

    else :
        print('Without joblib or multiprocessing libraries the calculation may be very long.')
        def xopt_solve (Xdetected) :
            Ns = Xdetected.shape[1]
            xopt = np.zeros((3*Ns))
            x0_part = x0
            for i in range (Xdetected.shape[1]) :
                X0i = Xdetected[:,i]
                x0i = x0_part[:,i]
                xopti, pcov = sopt.curve_fit(Soloff_Polynome({'polynomial_form' : Soloff_pform}).polynomial_LM_CF, 
                                            A0, 
                                            X0i, 
                                            p0 = x0i, 
                                            method ='lm')
                xopt[i], xopt[Ns + i], xopt[2*Ns + i] = xopti
            return (xopt)
        xopt = xopt_solve (Xdetected)
        xopt = xopt.reshape((3,N))

    Xcalculated = Soloff_Polynome({'polynomial_form' : Soloff_pform}).polynomial_system(xopt, A0)
    Xdiff = np.absolute(Xcalculated - Xdetected)
    print(str(Soloff_pform), ' : The max error between detected and calculated points is ', np.max(Xdiff), ' pixels.')
    
    return (xopt, Xcalculated, Xdetected)

def AI_solve_simultaneously (file,
                             n_estimators=800, 
                             min_samples_leaf=1, 
                             min_samples_split=2, 
                             random_state=1, 
                             max_features='sqrt',
                             max_depth=100,
                             bootstrap='true',
                             hyperparameters_tuning = False) :  
    """Calculation of the AI model between all inputs (Xl and Xr) and 
    outputs (x,y and z)
    
    Args:
       file : str
           Name of saving file for training
       n_estimators, 
       min_samples_leaf, 
       min_samples_split, 
       random_state, 
       max_features, 
       max_depth, 
       bootstrap,
       hyperparameters_tuning : 
          More information on the link :
          https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
           
           
    Returns:
       model : sklearn.ensemble._forest.RandomForestRegressor
           IA metamodel
       accuracy : int
           Accuracy of the IA metamodel compared with the training datas.
           
    """
    dat=pd.read_csv(file, sep=" " )
    dat=np.array(dat)
    # The model learn on 4/5 of all datas. Then the accuracy is estimated on 
    # the last 1/5 datas.
    N = int(len(dat)*4/5)
    
    # 1st meta-model
    X=dat[0:N,0:4]
    Y=dat[0:N,4:7]
    model = RandomForestRegressor(n_estimators=n_estimators, 
                                  min_samples_leaf=min_samples_leaf, 
                                  min_samples_split=min_samples_split, 
                                  random_state=random_state, 
                                  max_features=max_features,
                                  max_depth=max_depth,
                                  bootstrap=bootstrap)
    print('IA model training : x,y,z')
    model.fit(X,Y)
    
    # TEST 
    X2=dat[N:,0:4]
    Y2=dat[N:,4:7]
    
    # hyperparameter tuning 
    if hyperparameters_tuning :
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 200, 
                                                    stop = 2000, 
                                                    num = 10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]# Create the random grid
        random_grid = {'n_estimators': n_estimators,
                        'max_features': max_features,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf,
                        'bootstrap': bootstrap}
        
        rf_random = RandomizedSearchCV(estimator = model, 
                                       param_distributions = random_grid, 
                                       n_iter = 100, 
                                       cv = 3, 
                                       verbose = 2, 
                                       random_state = 42, 
                                       n_jobs = -1)
        rf_random.fit(X, Y)
        
        best_random = rf_random.best_estimator_
        print('Best hyper parameters')
        print(best_random)
    
    #################################
    def evaluate(model, test_features, test_labels):
        predictions = model.predict(test_features)
        errors = abs(predictions - test_labels)
        mape = 100 * np.mean(errors / np.max(test_labels))
        accuracy = 100 - mape
        print('Model Performance')
        print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
        print('Accuracy = {:0.2f}%.'.format(accuracy))
        return accuracy
    
    accuracy = evaluate(model, X2, Y2)
    return(model, accuracy)

def AI_solve_independantly (file,
                            n_estimators=800, 
                            min_samples_leaf=1, 
                            min_samples_split=2, 
                            random_state=1, 
                            max_features='sqrt',
                            max_depth=100,
                            bootstrap='true',
                            hyperparameters_tuning = False) :  
    """Calculation of the AI models between all inputs (Xl and Xr) and each 
    output (x,y or z)

    
    Args:
       file : str
           Name of saving file for training
       n_estimators, 
       min_samples_leaf, 
       min_samples_split, 
       random_state, 
       max_features, 
       max_depth, 
       bootstrap,
       hyperparameters_tuning : 
          More information on the link :
          https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
           
    Returns:
       modelx, modely, modelz : sklearn.ensemble._forest.RandomForestRegressor
           IA metamodel for x,y and z coordinates
       accuracyx, accuracyy, accuracyz : int
           Accuracy of the IA metamodel compared with the training datas
           for x,y and z coordinates.
           
    """
    dat=pd.read_csv(file, sep=" " )
    #Build correlation matrix
    import seaborn as sn
    corrMatrix = dat.corr()
    sn.heatmap(corrMatrix, annot=True)
    plt.show()
    
    dat=np.array(dat)
    # The model learn on 4/5 of all datas. Then the accuracy is estimated on 
    # the last 1/5 datas.
    N = int(len(dat)*4/5)
    
    # 1st meta-model
    X=dat[0:N,0:4]
    Yx=dat[0:N,4]
    Yy=dat[0:N,5]
    Yz=dat[0:N,6]
    modelx = RandomForestRegressor(n_estimators=n_estimators, 
                                    min_samples_leaf=min_samples_leaf, 
                                    min_samples_split=min_samples_split, 
                                    random_state=random_state, 
                                    max_features=max_features,
                                    max_depth=max_depth,
                                    bootstrap=bootstrap)
    
    modely = RandomForestRegressor(n_estimators=n_estimators, 
                                  min_samples_leaf=min_samples_leaf, 
                                  min_samples_split=min_samples_split, 
                                  random_state=random_state, 
                                  max_features=max_features,
                                  max_depth=max_depth,
                                  bootstrap=bootstrap)
    
    modelz = RandomForestRegressor(n_estimators=n_estimators, 
                                  min_samples_leaf=min_samples_leaf, 
                                  min_samples_split=min_samples_split, 
                                  random_state=random_state, 
                                  max_features=max_features,
                                  max_depth=max_depth,
                                  bootstrap=bootstrap)
    
    print('IA model training : x')
    modelx.fit(X,Yx)
    print('IA model training : y')
    modely.fit(X,Yy)
    print('IA model training : z')
    modelz.fit(X,Yz)
    
    # TEST 
    X2=dat[N:,0:4]
    Yx2=dat[N:,4]
    Yy2=dat[N:,5]
    Yz2=dat[N:,6]
    
    # hyperparameter tuning 
    if hyperparameters_tuning :
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 200, 
                                                    stop = 2000, 
                                                    num = 10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]# Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
        
        rf_randomx = RandomizedSearchCV(estimator = modelx, 
                                        param_distributions = random_grid, 
                                        n_iter = 100, 
                                        cv = 3, 
                                        verbose=2, 
                                        random_state=42, 
                                        n_jobs = -1)
        
        rf_randomy = RandomizedSearchCV(estimator = modely, 
                                        param_distributions = random_grid, 
                                        n_iter = 100, 
                                        cv = 3, 
                                        verbose=2, 
                                        random_state=42, 
                                        n_jobs = -1)
        
        rf_randomz = RandomizedSearchCV(estimator = modelz, 
                                        param_distributions = random_grid, 
                                        n_iter = 100, 
                                        cv = 3, 
                                        verbose=2, 
                                        random_state=42, 
                                        n_jobs = -1)
        
        rf_randomx.fit(X, Yx)
        rf_randomy.fit(X, Yy)
        rf_randomz.fit(X, Yz)
        
        best_randomx = rf_randomx.best_estimator_
        best_randomy = rf_randomy.best_estimator_
        best_randomz = rf_randomz.best_estimator_
        print('Best hyper parameters for x')
        print(best_randomx)
        print('Best hyper parameters for y')
        print(best_randomy)
        print('Best hyper parameters for z')
        print(best_randomz)
    
    #################################
    def evaluate(model, test_features, test_labels):
        predictions = model.predict(test_features)
        errors = abs(predictions - test_labels)
        mape = 100 * np.mean(errors / np.max(test_labels))
        accuracy = 100 - mape
        print('Model Performance')
        print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
        print('Accuracy = {:0.2f}%.'.format(accuracy))
        return accuracy
    
    accuracyx = evaluate(modelx, X2, Yx2)
    accuracyy = evaluate(modely, X2, Yy2)
    accuracyz = evaluate(modelz, X2, Yz2)

    return(modelx, 
           modely,
           modelz,
           accuracyx,
           accuracyy,
           accuracyz)

def AI_solve_zdependantly (file,
                           n_estimators=800, 
                           min_samples_leaf=1, 
                           min_samples_split=2, 
                           random_state=1, 
                           max_features='sqrt',
                           max_depth=100,
                           bootstrap='true',
                           hyperparameters_tuning = False) :  
    """Calculation of the AI models between all inputs (Xl and Xr) and each 
    output (x,y or z)
    
    Args:
       file : str
           Name of saving file for training
       n_estimators, 
       min_samples_leaf, 
       min_samples_split, 
       random_state, 
       max_features, 
       max_depth, 
       bootstrap,
       hyperparameters_tuning : 
          More information on the link :
          https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
           
           
    Returns:
       modelx, modely, modelz : sklearn.ensemble._forest.RandomForestRegressor
           IA metamodel for x,y and z coordinates
       accuracyx, accuracyy, accuracyz : int
           Accuracy of the IA metamodel compared with the training datas
           for x,y and z coordinates.
           
    """
    dat=pd.read_csv(file, sep=" " )
    #Build correlation matrix
    import seaborn as sn
    corrMatrix = dat.corr()
    sn.heatmap(corrMatrix, annot=True)
    plt.show()
    
    dat=np.array(dat)
    # The model learn on 4/5 of all datas. Then the accuracy is estimated on 
    # the last 1/5 datas.
    N = int(len(dat)*4/5)
    
    # 1st meta-model
    X=dat[0:N,0:4]
    Xp=dat[0:N,0:6]
    Yx=dat[0:N,4]
    Yy=dat[0:N,5]
    Yz=dat[0:N,6]
    modelx = RandomForestRegressor(n_estimators=n_estimators, 
                                    min_samples_leaf=min_samples_leaf, 
                                    min_samples_split=min_samples_split, 
                                    random_state=random_state, 
                                    max_features=max_features,
                                    max_depth=max_depth,
                                    bootstrap=bootstrap)
    
    modely = RandomForestRegressor(n_estimators=n_estimators, 
                                  min_samples_leaf=min_samples_leaf, 
                                  min_samples_split=min_samples_split, 
                                  random_state=random_state, 
                                  max_features=max_features,
                                  max_depth=max_depth,
                                  bootstrap=bootstrap)
    
    modelz = RandomForestRegressor(n_estimators=n_estimators, 
                                  min_samples_leaf=min_samples_leaf, 
                                  min_samples_split=min_samples_split, 
                                  random_state=random_state, 
                                  max_features=max_features,
                                  max_depth=max_depth,
                                  bootstrap=bootstrap)
    
    print('IA model training : x')
    modelx.fit(X,Yx)
    print('IA model training : y')
    modely.fit(X,Yy)
    print('IA model training : z')
    modelz.fit(Xp,Yz)
    
    # TEST 
    X2=dat[N:,0:4]
    X2p=dat[N:,0:6]
    Yx2=dat[N:,4]
    Yy2=dat[N:,5]
    Yz2=dat[N:,6]
    
    # hyperparameter tuning 
    if hyperparameters_tuning :
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 200, 
                                                    stop = 2000, 
                                                    num = 10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]# Create the random grid
        random_grid = {'n_estimators': n_estimators,
                        'max_features': max_features,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf,
                        'bootstrap': bootstrap}
        
        rf_randomx = RandomizedSearchCV(estimator = modelx, 
                                        param_distributions = random_grid, 
                                        n_iter = 100, 
                                        cv = 3, 
                                        verbose=2, 
                                        random_state=42, 
                                        n_jobs = -1)
        
        rf_randomy = RandomizedSearchCV(estimator = modely, 
                                        param_distributions = random_grid, 
                                        n_iter = 100, 
                                        cv = 3, 
                                        verbose=2, 
                                        random_state=42, 
                                        n_jobs = -1)
        
        rf_randomz = RandomizedSearchCV(estimator = modelz, 
                                        param_distributions = random_grid, 
                                        n_iter = 100, 
                                        cv = 3, 
                                        verbose=2, 
                                        random_state=42, 
                                        n_jobs = -1)
        
        rf_randomx.fit(X, Yx)
        rf_randomy.fit(X, Yy)
        rf_randomz.fit(Xp, Yz)
        
        best_randomx = rf_randomx.best_estimator_
        best_randomy = rf_randomy.best_estimator_
        best_randomz = rf_randomz.best_estimator_
        print('Best hyper parameters for x')
        print(best_randomx)
        print('Best hyper parameters for y')
        print(best_randomy)
        print('Best hyper parameters for z')
        print(best_randomz)
    
    #################################
    def evaluate(model, test_features, test_labels):
        predictions = model.predict(test_features)
        errors = abs(predictions - test_labels)
        mape = 100 * np.mean(errors / np.max(test_labels))
        accuracy = 100 - mape
        print('Model Performance')
        print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
        print('Accuracy = {:0.2f}%.'.format(accuracy))
        return accuracy
    
    accuracyx = evaluate(modelx, X2, Yx2)
    accuracyy = evaluate(modely, X2, Yy2)
    accuracyz = evaluate(modelz, X2p, Yz2)
    return(modelx, 
           modely,
           modelz,
           accuracyx,
           accuracyy,
           accuracyz)

def AI_HGBoost (file) :  
    """Calculation of the AI models between all inputs (Xl and Xr) and each 
    output (x,y or z)
    Args:
       file : str
          Name of saving file for training
          More information on the link :
          https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html#sklearn.ensemble.HistGradientBoostingRegressor
          
           
    Returns:
       modelx, modely, modelz : sklearn.ensemble._forest.RandomForestRegressor
           IA metamodel for x,y and z coordinates
       accuracyx, accuracyy, accuracyz : int
           Accuracy of the IA metamodel compared with the training datas
           for x,y and z coordinates.
           
    """
    dat=pd.read_csv(file, sep=" " )
    dat=np.array(dat)
    # The model learn on 4/5 of all datas. Then the accuracy is estimated on 
    # the last 1/5 datas.
    N = int(len(dat)*4/5)
    
    # 1st meta-model
    X=dat[0:N,0:4]
    Yx=dat[0:N,4]
    Yy=dat[0:N,5]
    Yz=dat[0:N,6]
    print('IA model training : x,y,z')
    modelx = HistGradientBoostingRegressor().fit(X, Yx)
    modely = HistGradientBoostingRegressor().fit(X, Yy)
    modelz = HistGradientBoostingRegressor().fit(X, Yz)
        
    # TEST 
    X2=dat[N:,0:4]
    Y2x=dat[N:,4]
    Y2y=dat[N:,5]
    Y2z=dat[N:,6]
    
    '''
    # hyperparameter tuning 
    if hyperparameters_tuning :
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 200, 
                                                    stop = 2000, 
                                                    num = 10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]# Create the random grid
        random_grid = {'n_estimators': n_estimators,
                        'max_features': max_features,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf,
                        'bootstrap': bootstrap}
        
        rf_random = RandomizedSearchCV(estimator = model, 
                                       param_distributions = random_grid, 
                                       n_iter = 100, 
                                       cv = 3, 
                                       verbose=2, 
                                       random_state=42, 
                                       n_jobs = -1)
        rf_random.fit(X, Y)
        
        best_random = rf_random.best_estimator_
        print('Best hyper parameters')
        print(best_random)
    '''
    #################################
    def evaluate(model, test_features, test_labels):
        predictions = model.predict(test_features)
        errors = abs(predictions - test_labels)
        mape = 100 * np.mean(errors / np.max(test_labels))
        accuracy = 100 - mape
        print('Model Performance')
        print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
        print('Accuracy = {:0.2f}%.'.format(accuracy))
        return accuracy
    
    accuracyx = evaluate(modelx, X2, Y2x)
    accuracyy = evaluate(modely, X2, Y2y)
    accuracyz = evaluate(modelz, X2, Y2z)
    
    model = [modelx, modely, modelz]
    accuracy = [accuracyx, accuracyy, accuracyz]
    
    return(model, accuracy) 
    
def AI_LinearRegression (file) :  
    """Calculation of the AI models between all inputs (Xl and Xr) and each 
    output (x,y or z)
    Args:
       file : str
          Name of saving file for training
          More information on the link :
          https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
          
           
    Returns:
       model = [modelx, modely, modelz] : list of sklearn.ensemble._forest.RandomForestRegressor
           IA metamodel for x,y and z coordinates
       accuracyx, accuracyy, accuracyz : int
           Accuracy of the IA metamodel compared with the training datas
           for x,y and z coordinates.
           
    """
    dat=pd.read_csv(file, sep=" " )
    dat=np.array(dat)
    # The model learn on 4/5 of all datas. Then the accuracy is estimated on 
    # the last 1/5 datas.
    N = int(len(dat)*4/5)
    
    # 1st meta-model
    X=dat[0:N,0:4]
    Yx=dat[0:N,4]
    Yy=dat[0:N,5]
    Yz=dat[0:N,6]
    print('IA model training : x,y,z')
    modelx = LinearRegression().fit(X, Yx)
    modely = LinearRegression().fit(X, Yy)
    modelz = LinearRegression().fit(X, Yz)
        
    # TEST 
    X2=dat[N:,0:4]
    Y2x=dat[N:,4]
    Y2y=dat[N:,5]
    Y2z=dat[N:,6]
    
    def evaluate(model, test_features, test_labels):
        predictions = model.predict(test_features)
        errors = abs(predictions - test_labels)
        mape = 100 * np.mean(errors / np.max(test_labels))
        accuracy = 100 - mape
        print('Model Performance')
        print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
        print('Accuracy = {:0.2f}%.'.format(accuracy))
        return accuracy
    
    accuracyx = evaluate(modelx, X2, Y2x)
    accuracyy = evaluate(modely, X2, Y2y)
    accuracyz = evaluate(modelz, X2, Y2z)
    
    model = [modelx, modely, modelz]
    accuracy = [accuracyx, accuracyy, accuracyz]
    
    return(model, accuracy) 

if __name__ == '__main__' :
    ()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt


class Direct_Polynome(dict) :
    def __init__(self, _dict_):
        self._dict_ = _dict_
        self.polynomial_form = _dict_['polynomial_form']

    def pol_form (self, Xl, Xr) :
        """Create the matrix M = f(Xl,Xr) with f the polynomial function of degree n
        
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

        return (M)
    

class Polynome(dict) :
    def __init__(self, _dict_):
        self._dict_ = _dict_
        self.polynomial_form = _dict_['polynomial_form']

    def pol_form (self, x) :
        """Create the matrix M = f(x) with f the polynomial function of degree (aab : a for x1, x2 and b for x3)
        
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
        if   polynomial_form == 111 :
            M = np.asarray ([np.ones((n)),   x1,        x2,        x3])
        elif polynomial_form == 221 :
            x12 = x1 * x1
            x22 = x2 * x2
            M = np.asarray ([np.ones((n)),   x1,        x2,        x3,         x12,
                             x1 *x2,         x22,       x1*x3,     x2*x3])   
        elif polynomial_form == 222 :
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
        elif polynomial_form == 333 :
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
        elif polynomial_form == 444 :
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
        elif polynomial_form == 555 :
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


    def polynomial_LM (self, a, *x) :
        polynomial_form = self.polynomial_form
        x = np.array ([x])
        x = x.reshape((3,len(x[0])//3))
        M = Polynome({'polynomial_form' : polynomial_form}).pol_form(x)    
        X = np.matmul(a, M)
        X = X.reshape(4*len(x[0]))
        return (X)
    
    def polynomial_Least_Square (self, x, X, a) :
        polynomial_form = self.polynomial_form
        x = np.array ([x])
        x = x.reshape((3,len(x[0])//3))
        M = Polynome({'polynomial_form' : polynomial_form}).pol_form(x) 
        Xc = np.matmul(a, M)
        Xc = Xc.reshape(4*len(x[0]))
        return (X-Xc)
    
    def polynomial_system (self, x, a) :
        """Create the matrix M = f(x) with f the polynomial function of degree (aab : a for x1, x2 and b for x3)
        
        Args:
           x : numpy.ndarray
               Real points x(x1, x2, x3)
           a : numpy.ndarray, opt
               cst of the polynomial function M = f(x)
           
        Returns:
           M : numpy.ndarray
               M = f(x)
        """
        polynomial_form = self.polynomial_form
        M = Polynome({'polynomial_form' : polynomial_form}).pol_form(x)   
        X = np.matmul(a, M)
            
        return(X)    

def fit_plan_to_points(point, 
                        ax = plt.subplot(111, projection='3d'), 
                        title = 'no title', 
                        axes_xyz = 1, 
                        label = 'xs'):
    """Create the matrix Y = f(x) with f the polynomial function chose
    
    Args:
       point : numpy.ndarray (shape = b,3)
           Real points x(x1, x2, x3)
       ax : ax = plt.subplot(111, projection='3d')
       title : str
       
    Returns:
       plot points + associated plan
    """
    if axes_xyz == 1 :
        xs = point[:,0]
        ys = point[:,1]
        zs = point[:,2]
    elif axes_xyz == 0 :
        xs, ys, zs = point 

    ax.scatter(xs, ys, zs, color='b', label = label)
    
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
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    X,Y = np.meshgrid(np.linspace(xlim[0], xlim[1], 10),
                      np.linspace(ylim[0], ylim[1], 10))
    Z = np.zeros(X.shape)
    for r in range(X.shape[0]):
        for c in range(X.shape[1]):
            Z[r,c] = fit[0] * X[r,c] + fit[1] * Y[r,c] + fit[2]
    ax.plot_wireframe(X,Y,Z, color='k')
    
    ax.set_title(title)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_zlabel('z (mm)')

    return (errors, mean_error, residual)

def fit_plans_to_points(points, 
                         title = 'no title', 
                         axes_xyz = 2, 
                         label = 'xs'):
    """Create the matrix Y = f(x) with f the polynomial function chose
    
    Args:
       points : numpy.ndarray (shape = a,b,3)
           Real points x(x1, x2, x3)
       title : str
       
    Returns:
       plot points + associated plans on an unique graph
    """
    # plot raw data
    plt.figure()
    ax = plt.subplot(111, projection='3d')
    l, m, n = points.shape
    errors = np.zeros((l, n))
    print(errors.shape)

    mean_error = np.zeros(l)
    residual = np.zeros(l)
    for i in range (len(points)) :
        point = points[i]
        errors[i], mean_error[i], residual[i] = fit_plan_to_points(point, ax, title = title, axes_xyz = axes_xyz-1, label = label)
        
    print('Plan square max error = ', (np.max(errors)), ' mm')
    print('Plan square mean error = ', (np.mean(mean_error**2))**(1/2), ' mm')
    print('Plan square mean residual = ', (np.mean(residual**2))**(1/2), ' mm')

    plt.show()

def refplans(xc1, x3_list) :
    m, n = xc1.shape
    o = len(x3_list)
    n = n//o
    x,y,z = xc1
    xcons = np.zeros((o,m,n))
    for i in range (o) :
        xcons[i] = x[i*n:(i+1)*n], y[i*n:(i+1)*n], z[i*n:(i+1)*n]
    fit_plans_to_points(xcons, 
                        title = 'Calibration plans',
                        axes_xyz = 1,
                        label = 'xs')

if __name__ == '__main__' :
    ()





import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.stats import skew, kurtosis
from numpy import linalg as LA
from glob import glob
from os import chdir

default = dict(
    alpha = 3,
    delta=1,
    gamma=0,
    fpi=20,
    sor=10,
    omega=1.6,
    nstages=100,
    minsize=1000, # Y-Size of the first estimation (Lower is minsize, higher is the accuracy but longer is the calculation
    rfactor=.95,
    interp_d=cv2.INTER_AREA, # When reducing
    interp_u=cv2.INTER_CUBIC, # When enlarging
    progress=True
  )

class Vref:
  """
  Coarse-to-fine variational refinement using OpenCV
  """
  def __init__(self,**kwargs):
    self.vr = cv2.VariationalRefinement_create()
    for k,v in default.items():
      setattr(self,k,kwargs.pop(k,v))
    if kwargs:
      raise AttributeError(f"Unknown parameter(s):{kwargs}")
    self.h = self.w = -1

  @property
  def alpha(self):
    return self.vr.getAlpha()

  @alpha.setter
  def alpha(self,v):
    self.vr.setAlpha(v)

  @property
  def delta(self):
    return self.vr.getDelta()

  @delta.setter
  def delta(self,v):
    self.vr.setDelta(v)

  @property
  def gamma(self):
    return self.vr.getGamma()

  @gamma.setter
  def gamma(self,v):
    self.vr.setGamma(v)

  @property
  def fpi(self):
    return self.vr.getFixedPointIterations()

  @fpi.setter
  def fpi(self,v):
    self.vr.setFixedPointIterations(v)

  @property
  def sor(self):
    return self.vr.getSorIterations()

  @sor.setter
  def sor(self,v):
    self.vr.setSorIterations(v)

  @property
  def omega(self):
    return self.vr.getOmega()

  @omega.setter
  def omega(self,v):
    self.vr.setOmega(v)

  def calc_pyramid(self):
    """
    Computes the shape of the image on all the levels
    """
    self.shapelist = []
    f = 1
    while len(self.shapelist) < self.nstages \
        and min(self.h,self.w)*f >= self.minsize:
      self.shapelist.append((int(round(self.h*f)),int(round(self.w*f))))
      f *= self.rfactor
    if self.progress:
      self.total = sum([i*j for i,j in self.shapelist])
        
  def resample_img(self,ima,imb):
    """
    Makes lists of the images at every resolution of the pyramid
    """
    self.imalist = [ima]
    self.imblist = [imb]
    for y,x in self.shapelist[1:]:
      self.imalist.append(cv2.resize(self.imalist[-1],(x,y),
        interpolation=self.interp_d))
      self.imblist.append(cv2.resize(self.imblist[-1],(x,y),
        interpolation=self.interp_d))

  def print_progress(self,erase=True):
    print(("\r" if erase else "") +
        "{:.2f} %".format(self.processed/self.total*100),end="",flush=True)

  def calc(self,ima,imb,f=None):
    """
    Compute the variational refinement with the coarse-to-fine approach
    """
    if ima.shape != (self.h,self.w):
      self.h,self.w = ima.shape
      self.calc_pyramid()
    # Prepare all the images
    self.resample_img(ima,imb)
    if f is None: # No field given, start from 0
      f = np.zeros(self.shapelist[-1]+(2,),dtype=np.float32)
    else: # Resample it to the resolution of the first level

        f = cv2.resize(f,self.shapelist[-1][::-1],
          interpolation=self.interp_d)*self.rfactor**len(self.shapelist)
    # Compute the first field
    self.vr.setAlpha(self.vr.getAlpha()*self.shapelist[-1][0]/self.shapelist[0][0])
    self.vr.setDelta(self.vr.getDelta()*self.shapelist[0][0]/self.shapelist[-1][0])
    print("\ni=",0)
    print("alpha=",self.vr.getAlpha())
    print("delta=",self.vr.getDelta())
    print("shape=",self.shapelist[-1])
    self.vr.calc(self.imalist[-1],self.imblist[-1],f)
    #residue=getresidue(self.imalist[-1],self.imblist[-1],f)
    #plt.figure()
    #plt.imshow(residue, cmap='gray', vmin=-50, vmax=50)
    if self.progress:
      i,j = self.shapelist[-1]
      self.processed = i*j
      self.print_progress(False)

    # Working our way up the pyramid (skipping the lowest)
    for i,(ima,imb,shape) in enumerate(list(zip(
        self.imalist,self.imblist,self.shapelist))[-2::-1]):
        f = cv2.resize(f,shape[::-1],interpolation=self.interp_u)/self.rfactor
        self.vr.setAlpha(self.vr.getAlpha()*self.shapelist[len(self.shapelist)-i-2][0]/self.shapelist[len(self.shapelist)-i-1][0])
        self.vr.setDelta(self.vr.getDelta()*self.shapelist[len(self.shapelist)-i-1][0]/self.shapelist[len(self.shapelist)-i-2][0])
        print("\n\ni=",i+1)
        print("alpha=",self.vr.getAlpha())
        print("delta=",self.vr.getDelta())
        print("shape=",self.shapelist[len(self.shapelist)-i-2])
        self.vr.calc(ima,imb,f)
        #residue=getresidue(self.imalist[len(self.imalist)-i-2],self.imblist[len(self.imblist)-i-2],f)
        #plt.figure()
        #plt.imshow(residue, cmap='gray', vmin=-50, vmax=50)
        if self.progress:
            self.processed += shape[0]*shape[1]
            self.print_progress()
    return f

#variational refinement
def getresidue(ima,
               imb,
               f): 
    x, y = np.meshgrid(np.arange(ima.shape[1]), np.arange(ima.shape[0])) 
    x = x.astype('float32') 
    y = y.astype('float32') 
    remap=cv2.remap(imb,x+f[:,:,0],y+f[:,:,1],cv2.INTER_LINEAR) 
    residue=ima-remap.astype('float32') 
    return residue

def displacement_field (Im1, 
                        Im2, 
                        window = [False],
                        vr_kwargs=dict()) :
    """Calcul the displacement field between two images.

    Args:
        image_1 : type = class 'str'
            reference image      
        image_2 : type = class 'str'
            image you want to compare with the image_2
        window : type = list
            The ZOI of the images
        vr_kwargs : type = dict
            Correlations parameters

    Returns:
        U : type = numpy.ndarray
            U displacement field (x coord)
        V : type : numpy.ndarray
            V displacement field (y coord)
        
    """    
    # Check the shape of the image sequence
    if(Im1.shape != Im2.shape):
        raise ValueError("Images must be the same shape")
        
    if any(window) :
        [lx1, lx2], [ly1, ly2] = window
        Im1 = Im1[ly1:ly2, lx1:lx2]
        Im2 = Im2[ly1:ly2, lx1:lx2]
        

    alpha = vr_kwargs['alpha'] if 'alpha' in vr_kwargs else 3
    delta = vr_kwargs['delta'] if 'delta' in vr_kwargs else 1
    gamma = vr_kwargs['gamma'] if 'gamma' in vr_kwargs else 0
    iterations = vr_kwargs['iterations'] if 'iterations' in vr_kwargs else 10
    flow = cv2.DISOpticalFlow_create()
    flow.setFinestScale(0)
    # cf https://www.mia.uni-saarland.de/Publications/brox-eccv04-of.pdf
    flow.setVariationalRefinementAlpha(alpha)
    flow.setVariationalRefinementDelta(delta)
    flow.setVariationalRefinementGamma(gamma)
    flow.setVariationalRefinementIterations(iterations)
    res = flow.calc(Im1, Im2, None)

    U = res[:,:,0] # The X-displacement
    V = res[:,:,1] # The Y-displacement

    return (U,V)

def strain_fields (U, 
                   V,
                   W) :
    """Calcul all the strain fields between two images.

    Args:
        U : type = numpy.ndarray
            U displacement field (x coord)
        V : type : numpy.ndarray
            V displacement field (y coord)
        W : type : numpy.ndarray
            W displacement field (z coord)

    Returns:
        Exx : type = numpy.ndarray
            Exx strain field (dx/x)
        Exy : type = numpy.ndarray
            Exx strain field (dx/y)
        Eyx : type = numpy.ndarray
            Exx strain field (dy/x)
        Eyy : type = numpy.ndarray
            Exx strain field (dy/y)        
        Ezx : type = numpy.ndarray
            Exx strain field (dz/x)
        Ezy : type = numpy.ndarray
            Exx strain field (dz/y)          
    """  
    Exy,Exx = np.gradient(U)
    Eyy,Eyx = np.gradient(V)
    Ezy,Ezx = np.gradient(W)
    
    return (Exx, Exy, Eyx, Eyy, Ezx, Ezy)
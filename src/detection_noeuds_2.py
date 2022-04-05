import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches

from skimage.feature import structure_tensor, hessian_matrix
from skimage.filters import difference_of_gaussians, threshold_otsu
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
# from skimage.exposure import equalize_adapthist
# import math
# import glob

plt.close("all")
#plt.ion()
pix = 5

img= plt.imread('3.0.png')
H11, H22, H12 = hessian_matrix(img, 9)
invar = abs(H22)
thresh = threshold_otsu(invar)
bin_im = invar>thresh
label_img=label(clear_border(bin_im))
regions = regionprops(label_img)

name = []
boundbox = []
barx = []
bary = []
for i, region in enumerate(regions):
    if region.area>500:
        name.append(region.area)
        boundbox.append(region.bbox)
        barx.append(region.centroid[0])
        bary.append(region.centroid[1])
all_px=np.vstack([barx])
all_py=np.vstack([bary])
time = [0]
fig, ax = plt.subplots()
ax.imshow(img, cmap='gray')
for i in range(1,len(boundbox)): # for each ZOI
    gx, gy = barx[i], bary[i] # barycenter coordinate of each ZOI
    ax.plot(gy,gx, 'ro', markersize=2)

#t =np.array([0, 0])
#for j in np.arange(0, 1200, 1):
    ##if j >920:
        ##pix = 12
    ##else:
        ##pass
    #print(j)
    #image = plt.imread(Liste_image[j])
    
    #t = np.vstack([t , [float(Liste_image[j][66:-5]), int(Liste_image[j][59:65])]])
    #fig, ax = plt.subplots()
    #ax.imshow(image, cmap='gray')
    
    #for i in range(len(name)):
        #'''
        #Using previous image information, we loop over every previous regions.
        #Previous bbox are increased and binarized to detect the main spot.  
        #'''
        #if math.isnan(boundbox[i][0]):
            #'''
            #if a spot has disappeared (out of the frame...)
            #barycenter and bbox are set to 'nan' . They are excluded from 
            #computation at this step.
            #New values are set to 'nan' as well
            #'''
            ##print (i, 'NANAN')
            #barx[i] = np.float('nan')
            #bary[i] = np.float('nan')
            #boundbox[i] = (np.float('nan'),np.float('nan'),
                    #np.float('nan'),np.float('nan'))
        #else :
            #minr, minc, maxr, maxc = boundbox[i] # bbox of each ZOI
            #if minr < pix:
                #minr = pix
            #if minc < pix:
                #minc = pix
            #imette = -difference_of_gaussians(
                #image[minr-pix:maxr+pix, minc-pix:maxc+pix], 2.9, 3)
            ##imette = abs(255 - image[minr-pix:maxr+pix, minc-pix:maxc+pix])
            #invar_ZOI = invariant(imette, 5)
            ##invar_ZOI = imette
            #thresh = threshold_otsu(invar_ZOI)
            #invar_ZOI = invar_ZOI>thresh
            #label_img = label(255.*invar_ZOI)
            #regions = regionprops((label_img))
            #area = [region.area for region in regions]
            ##if i==68:
                ##print('ok')
                ##print(minr, minc, maxr, maxc)
                ##plt.figure()
                ##plt.imshow(invar_ZOI)
                ##plt.figure()
                ##plt.imshow(image[minr-pix:maxr+pix, minc-pix:maxc+pix])
                ####rect = mpatches.Rectangle((minc, minr), maxc - (minc), maxr - (minr),
                                    ##fill=False, edgecolor='red', linewidth=2)
                ##ax.add_patch(rect)
            #if area:
                #'''
                #if spots are detected in the ZOI, the bigger one is selected.
                #Barycenters and bounding box of this spot are then updated
                #'''
                #roi_index = np.where(area==max(area))[0][0]
                #px,py = regions[roi_index].centroid # X,Y coordin. in local ZOI
                #ppx = minr-pix+px # compute X coordinate in global image
                #ppy = minc-pix+py # compute Y coordinate in global image
                #minrr, mincc, maxrr, maxcc = regions[roi_index].bbox
                #boundbox[i] = (minrr+minr-pix, mincc+minc-pix,
                                #maxrr+minr-pix, maxcc+minc-pix) # update bbox
                #barx[i] = ppx # update X barycenter coordinate
                #bary[i] = ppy # update Y barycenter coordinate
            #else:
                #''' 
                #if no spot is detected, boundbox and barycenter coordinates 
                #are set to 'nan'
                #'''
                #boundbox[i]=(np.float('nan'),np.float('nan'),\
                    #np.float('nan'),np.float( 'nan'))
                #barx[i]=np.float('nan')
                #bary[i]=np.float('nan')
            #ax.plot(ppy,ppx,'bo',markersize=1)
    ##plt.ylim(ymin=max(limits[1])+300, ymax=min(limits[1])-300) 
    ##plt.xlim(xmin=limits[0][0]-300, xmax=limits[0][-1]+300)
    #plt.savefig(Path + 'images_noeuds/img_%06d.png'%j,dpi=150)
    #plt.close()
    #all_px = np.vstack([all_px, barx]) 
    ### add updated X coord of all ZOI to previous ones
    #all_py = np.vstack([all_py, bary]) 
    ### add updated Y coord of all updated ZOI to previous ones
    ###print j  #indicator of advancement

#np.savetxt(Path + 'coord_px.txt', all_px) # save X
#np.savetxt(Path + 'coord_py.txt', all_py) # save Y
#np.savetxt(Path + 'time.txt', t)

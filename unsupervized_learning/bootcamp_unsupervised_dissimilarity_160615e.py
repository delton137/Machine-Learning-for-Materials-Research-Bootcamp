# Dissimilarity Analysis
# Required:
# data file unsupervised-1.mat
# images: oval_V, oval_H, rectangular_H
# Hu_moment invariants

# Load data
import scipy.io as sio
import numpy.matlib as nm
import matplotlib.pyplot as plt

# Loads data from FeGaPd composition spread
t = sio.loadmat('unsupervised-1.mat')
cmp = t['CMP']
tth = t['TTH']
xrd = t['XRD']

tth_ = nm.repmat(tth,xrd.shape[0],1)
#%%
# Plot data
# plot all diffraction patterns at once.

plt.figure(1)
plt.clf()
plt.subplot(3,1,1)
plt.plot(tth_.T, xrd.T)
plt.xlabel(r'2$\theta$')

#%%
# plot heatmap of diffraction patterns
plt.subplot(3,1,2)
plt.pcolor(xrd)

#%%
# plot one diffraction pattern.
plt.subplot(3,1,3)
plt.plot(tth_[0,],xrd[49,])
plt.xlabel(r'2$\theta$')


#%%  ------- Measures -----------------------

# Create data sets for Scale Invariance example
# create data sets 1 and 2
xrd1 = xrd[189:196,] # data set 1
xrd2 = xrd[nm.arange(6,-1,-1),] # data set 2

#%%
# plot data set 1
plt.figure(2)
plt.clf()
plt.subplot(2,1,1)
plt.plot(tth_[0:7,].T, xrd1.T);
plt.title('Data Set 1');
plt.xlabel(r'2$\theta$');

#%%
plt.subplot(2,1,2)
plt.plot(tth_[0:7,].T, xrd2.T)
plt.title('Data Set 2');
plt.xlabel(r'2$\theta$');
plt.tight_layout()
#%%
# Compute dissimilarity matrices

s = nm.concatenate((xrd1, xrd2), axis=0) # Combine data into one matrix.
from sklearn.metrics.pairwise import pairwise_distances

# generate dissimilarity matrix for L1, "cityblock" or "taxicab"

d1 = 


#%% Visualize Dissimilarity matrix

# City Block (L1)
plt.figure(3)
plt.clf()
plt.subplot(1,4,1)
plt.pcolor(d1)
plt.gca().invert_yaxis()
plt.title('City Block (L1)')

#%%
# generate dissimilarity matrix for L2, "Euclidean"


# generate dissimilarity matrix for cosine metric.
# !!! D3 = cosine dissimilarity!

# look at pdist options.

#%% Visualize Dissimilarity matrices

# Euclidean
plt.subplot(1,4,2)
plt.pcolor(d2)
plt.gca().invert_yaxis()
plt.title('Euclidean (L2)')

# And Cosine
plt.subplot(1,4,3)
plt.pcolor(d3)
plt.gca().invert_yaxis()
plt.title('Cosine')

#%% Visualize Dissimilarity matrix for nonordered data

# randomly permute the numbers from 1 to 14.
p = nm.random.permutation(14)

# compute the dissimilarity matrix for permuted data.
d4 = pairwise_distances( s[p,:] , metric='cosine')

# Visualize Dissimilarity matrix
plt.subplot(1,4,4)
plt.pcolor(d4)
plt.gca().invert_yaxis()
plt.title('Unordered data')



#%% Dissimilarity in feature space - Moment Invariants
from scipy import misc

# load the images
im1 = misc.imread('Oval_H.png')
im2 = misc.imread('Oval_V.png')
im3 = misc.imread('Rectangular_H.png')

#%% plot the images
plt.figure(1)
plt.clf()
plt.subplot(1,3,1)
plt.imshow(im1)

plt.subplot(1,3,2)
plt.imshow(im2)

plt.subplot(1,3,3)
plt.imshow(im3)

#%% convert the images into binary masks
# 1 inside shape, 0 outside
mask1 = im1[:,:,1] < 255;
mask2 = im2[:,:,1] < 255;
mask3 = im3[:,:,1] < 255;
mask = nm.dstack((1.*mask1,1.*mask2,1.*mask3))
# plot the masks
plt.figure(2)
plt.clf()
plt.subplot(1,3,1)
plt.imshow(mask[:,:,0])

plt.subplot(1,3,2)
plt.imshow(mask[:,:,1])

plt.subplot(1,3,3)
plt.imshow(mask[:,:,2])

#%% compute the absolute moment invariants
# uses code from scikit-image
# http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.moments_hu
import skimage.measure

hu = nm.zeros((3,7))
for i in range(0,3):
    nu = skimage.measure.moments_central(mask[:,:,i],125.5,125.5)
    nu = skimage.measure.moments_normalized(nu)
    hu[i,:] = skimage.measure.moments_hu(nu)
    
ami = nm.zeros((3,2))
ami[:,0] = 2./hu[:,0]
# ami[:,1] = 4./( hu[:,0]**2 - hu[:,1] )
ami[:,1] = 4./( nm.power(hu[:,0],2) - hu[:,1] )

#%%
bg = misc.imread('moment_invariant_plot.png')

plt.figure(3)
plt.clf()

# plot the moment invariant figure as a background 
plt.imshow(nm.flipud(bg), origin='lower')

#%% plot the ovals using moment invariants.
plt.plot(ami[0:1,0] * 310/8,ami[0:1,1]*300/100,'ro')

# plot the rectangle using moment invariants.
plt.plot(ami[2,0] * 310/8,ami[2,1]*300/100,'bo')
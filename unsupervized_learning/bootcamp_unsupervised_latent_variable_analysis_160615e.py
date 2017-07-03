#%% ----------- Spectral Umixing: Non-negative Matrix Factorization --------------------
# Load data
import scipy.io as sio
import numpy.matlib as nm
import matplotlib.pyplot as plt

t = sio.loadmat('unsupervised-1.mat')
cmp = t['CMP']
tth = t['TTH']
xrd = t['XRD']
tth_ = nm.repmat(tth,xrd.shape[0],1)

#%% Plot the data
 
plt.figure(1)
plt.clf()
plt.subplot(2,1,1)
plt.plot(tth_[0:5,:].T, xrd.T)
plt.xlabel(r'2$\theta$')


#%% Run NMF
from sklearn.decomposition import NMF
from random import randint
k = 7 # number of Endmembers
model = NMF(n_components=k, init='random', random_state=randint(0, 1000))
endmembers = model.fit(xrd).components_ # X in our equation
abundances = model.transform(xrd) # beta in our equation

# plot the results
plt.figure(1)
plt.subplot(2,1,2)
plt.cla()
plt.plot(tth_[0:5,:].T, endmembers.T)
plt.xlabel(r'2$\theta$')

# Run these steps a few times until you see a significant change.
# The endmembers change on each run!

#%% plot the various endmembers
plt.figure(2)
plt.clf()

for i in range(0,k):
   plt.subplot(k,1,i+1)
   plt.plot(tth_[0,:], endmembers[i,:])
   plt.title('E' + str(i+1))
plt.tight_layout()
# Each person may get different results!
# This is because NMF gives a local minima solution.

#%% plot the abundances for diffraction pattern 50
plt.figure(3)
plt.clf()
labels = ['E1','E2','E3','E4','E5','E6','E7']
plt.pie(abundances[49,:],labels = labels)

# Each person may get different results!
# This is because NMF gives a local minima solution.




#%% ----------- Latent Variable Analysis: Principal Component Analysis (PCA) ----------------------------------
# compute PCA
from sklearn.decomposition import PCA

# subtract mean
xrd_mean = xrd.mean(axis=0)
xrd_ = xrd - nm.repmat(xrd_mean,xrd.shape[0],1)

# Run PCA

loading_vectors = pca.components_
explained_variance = pca.explained_variance_
principal_components = pca.transform(xrd_)

#%% Compute PCA-based approximations for smoothing

N = 50
x = xrd[N-1,]

plt.figure(1)
plt.clf()
plt.subplot(4,1,1)
plt.plot(tth.flatten(),xrd[N-1,])
plt.title('Original')
plt.xlabel(r'2$\theta$')

#%%
# approximation with first 10 loading vectors
# Following equation given in slides
lv = loading_vectors[0:10,]
xa10 = nm.dot(principal_components[N-1,0:10], lv) + xrd_mean

# Plot approximation for 10 loading vectors
plt.subplot(4,1,2)
plt.plot(tth.flatten(), xa10)
plt.xlabel(r'2$\theta$')
plt.title('10 loading vectors')

#%%

# approximation with first 50 loading vectors
lv = loading_vectors[0:50,]
xa50 = nm.dot(principal_components[N-1,0:50], lv) + xrd_mean

# approximation with first 100 loading vectors
lv = loading_vectors[0:100,]
xa100 = nm.dot(principal_components[N-1,0:100], lv) + xrd_mean

#%%
# plot approximation for 50 PCs
plt.subplot(4,1,3)
plt.plot(tth.flatten(), xa50)
plt.xlabel(r'2$\theta$')
plt.title('50 loading vectors')

# plot approximation for 100 PCs
plt.subplot(4,1,4)
plt.plot(tth.flatten(), xa100)
plt.xlabel(r'2$\theta$')
plt.title('100 loading vectors')

plt.tight_layout()
#%% Investigate left over information

# compute remaining information for each approximation
xa10_remaining = nm.dot(principal_components[N-1,10:], loading_vectors[10:,])

xa50_remaining = nm.dot(principal_components[N-1,50:],  loading_vectors[50:,]) 

xa100_remaining = nm.dot(principal_components[N-1,100:],  loading_vectors[100:,])


# plot the left over information
plt.figure(2)
plt.clf()
plt.subplot(4,1,2)
plt.plot(tth.flatten(), xa10_remaining)
plt.xlabel(r'2$\theta$')
plt.ylim([-25, 310])

plt.subplot(4,1,3);
plt.plot(tth.flatten(), xa50_remaining)
plt.xlabel(r'2$\theta$')
plt.ylim([-25, 310])

plt.subplot(4,1,4);
plt.plot(tth.flatten(), xa100_remaining)
plt.xlabel(r'2$\theta$')
plt.ylim([-25, 310])

plt.tight_layout()

#%% plot the variance per variable for each loading vector
plt.subplot(4,1,1)
plt.plot(nm.arange(1,41),explained_variance[0:40]/tth_.shape[1],'b.')
plt.xlabel('loading vector number')
plt.title('explained variance / # of variables')

plt.tight_layout()

#%% Visualizing data in lower dimension

# create data sets 1 and 2
xrd1 = xrd[189:196,:] # data set 1
xrd2 = xrd[0:7,:] # data set 2

# Combine data into one matrix.
s = nm.concatenate((xrd1, xrd2), axis=0) 

#%% Plot data sets
plt.figure(1)
plt.clf()
plt.subplot(3,1,1)
plt.plot(tth_[0:7,:].T, xrd1.T)
plt.xlabel(r'2$\theta$')

plt.subplot(3,1,2)
plt.plot(tth_[0:7,:].T, xrd2.T)
plt.xlabel(r'2$\theta$')

#%% Run PCA on this data set
# compute PCA for XRD (from line 27)
# !!!!
s_mean = s.mean(axis=0)
s_ = s - nm.repmat(s_mean,s.shape[0],1)


loading_vectors = pca.components_
explained_variance = pca.explained_variance_
principal_components = pca.transform(s_)


#%% plt the variance/#variables for each loading vector

plt.subplot(3,1,3)
plt.plot(nm.arange(1,15),explained_variance/tth_.shape[1],'b.')
plt.xlabel('loading vector number')
plt.title('explained variance / # of variables')
plt.tight_layout()


#%% Plot the data in 2D
fig, ax = plt.subplots()
# plot class 1
plt.plot(principal_components[0:7,0], principal_components[0:7,1],'b.')

# plot class 2
plt.plot(principal_components[7:14,0], principal_components[7:14,1],'r.')

# set the x-spine 
ax.spines['left'].set_position('zero')

# turn off the right spine/ticks
ax.spines['right'].set_color('none')
ax.yaxis.tick_left()

# set the y-spine
ax.spines['bottom'].set_position('zero')

# turn off the top spine/ticks
ax.spines['top'].set_color('none')
ax.xaxis.tick_bottom()

plt.tight_layout()

#%% Visualize the loading vectors

k = 5
plt.figure(3)
plt.clf()
# plot each loading vector
for i in range(0,k):
   plt.subplot(k,1,i+1)
   plt.plot(tth_[0,:], loading_vectors[i,:])
   plt.title('u' + str(i+1) + ', variance/(#variables) = ' + str(explained_variance[i]/tth_.shape[1]))
plt.tight_layout()




#%% Compare PCA loading vectors to NMF Endmembers
plt.figure(1)
plt.clf()
plt.subplot(2,1,1)
plt.plot(tth_[0,:], loading_vectors[0,:],'b')
plt.plot(tth_[0,:], loading_vectors[1,:],'r')
plt.xlabel(r'2$\theta$')
plt.title('PCA loading vectors 1 & 2')

#%%

endmembers = model.fit(s).components_
abundances = model.transform(s)
plt.subplot(2,1,2)
plt.cla()
plt.plot(tth_[0,:], endmembers[0,:],'b')
plt.plot(tth_[0,:], endmembers[1,:],'r')
plt.xlabel(r'2$\theta$')
plt.title('NMF endmembers 1 & 2')
plt.tight_layout()

#%% PCA for outlier detection
plt.figure(5)
plt.clf()
plt.plot(tth_[0,],s[13,])
plt.xlabel(r'2$\theta$')
plt.tight_layout()



#%% ----------- Multidimensional Data Scaling --------------------------
# Look at multidimensional data scaling of cosine dissimilarity

from sklearn.manifold import MDS
from sklearn.metrics.pairwise import pairwise_distances

d = pairwise_distances(s, metric = 'cosine')



# MDS of 'D' in 2D
plt.figure(1)
plt.clf()
plt.plot(pos[0:7,0],pos[0:7,1],'bo')
plt.plot(pos[8:15,0],pos[8:15,1],'ro')
plt.title('MDS 2D mapping')
plt.tight_layout()


#%%
d_MDS = pairwise_distances(pos, metric = 'euclidean')

plt.figure(2)
plt.clf()
plt.subplot(1,2,1)
plt.pcolor(d)
plt.gca().invert_yaxis()
plt.title('cosine metric on XRD')
plt.colorbar()
plt.subplot(1,2,2)
plt.pcolor(d_MDS)
plt.gca().invert_yaxis()
plt.title('MDS mapped dissimilarity')
plt.colorbar()

plt.tight_layout()

#%% MDS for full XRD
d = pairwise_distances(xrd, metric = 'cosine')
mds = MDS(n_components=2, dissimilarity="precomputed")
pos = mds.fit(d).embedding_
d_MDS = pairwise_distances(pos, metric = 'euclidean')

plt.figure(2)
plt.clf()
plt.subplot(1,2,1)
plt.pcolor(d)
plt.gca().invert_yaxis()
plt.title('cosine metric on XRD')
plt.colorbar()
plt.subplot(1,2,2)
plt.pcolor(d_MDS)
plt.gca().invert_yaxis()
plt.title('MDS mapped dissimilarity')
plt.colorbar()
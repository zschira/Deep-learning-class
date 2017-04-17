from sklearn import decomposition
from sklearn.externals import joblib
import h5py
import os
import numpy as np


path = '/media/zach/AOP-NEON1-4/D17/SJER/2013/SJER_L1/SJER_Spectrometer/Reflectance/'
files = os.listdir(path)
file = path + files[0]
f = h5py.File(file)
dset = f["/Reflectance"]

y_start = 6015
arr = dset[56,y_start:y_start + 600,:]
inds = np.array(np.where(arr != 15000))
new_row = np.array(np.array(np.where(np.diff(inds[0,:]) == 1))) +1
start_inds = inds[1, new_row]
end_inds = inds[1, new_row - 1]
x1 = np.amax(start_inds)
x2 = np.amin(end_inds)
X = dset[:,y_start:y_start + 600,x1:x2]

def unravel(arr):
	img_shape = arr.shape
	X = np.zeros((img_shape[1]*img_shape[2], img_shape[0]))
	for i in range(img_shape[2]):
		X[img_shape[1]*i:img_shape[1]*i+img_shape[1], :] = np.transpose(arr[:,:,i])
	return X

X = unravel(X)

def featureNorm(X, lidar=False):
	if(lidar):
		mu = np.mean(X)
		sigma = np.std(X)
		X = (X - mu)/sigma
		return X
	X_shape = X.shape
	X_norm = X
	mu = np.mean(X, axis=1)
	sigma = np.std(X, axis = 1)
	for i in range(X_shape[1]):
		X_norm[:,i] = (X[:,i] - mu[i])/sigma[i]
	return X_norm

X = featureNorm(X)

def reduceDims(X):
	n_components = 2
	pca = decomposition.PCA(n_components=n_components)
	sample = np.random.randint(X.shape[0], size=100000)
	pca.fit(X[sample,:])
	return pca

pca = reduceDims(X)
joblib.dump(pca, 'PCA_fit.pkl')
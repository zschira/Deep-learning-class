 #Script to processraw data to pass to fast rcnn
import math
import numpy as np 
import h5py
import os
from sklearn import decomposition
from sklearn.externals import joblib
from sklearn.cluster import KMeans
import scipy.misc
import re
import gdal
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import xml.etree.cElementTree as ET

def getTarget(df):
	species = df['taxonid']
	labels, levels = pd.factorize(species)
	y = labels
	rad = df['canopydiam_90deg']
	return y, rad/2, levels

def getPlots(X):
	file = '../raw-data/plots.csv'
	df = pd.read_csv(file, skiprows = 0)
	east = df['easting']
	north = df['northing']
	plots = np.vstack((east,north))
	labels = np.zeros(len(X[0,:]))
	for i in range(len(east)):
		temp1 = X[0,:]-east[i]
		temp2 = X[1,:]-north[i]
		labels[np.logical_and(temp1 <= 20, temp2 <= 20)] = i
	return plots, labels



def cleanData(f, coords, check):
	dset = f['Reflectance']
	shape = dset.shape
	map_info = f['/map info'][0].decode().split(',')
	y_start = 50
	easting = float(map_info[3])
	northing = float(map_info[4])
	arr = dset[56,y_start:y_start + 600,:]
	inds = np.array(np.where(arr != 15000))
	new_row = np.array(np.array(np.where(np.diff(inds[0,:]) == 1))) +1
	start_inds = inds[1, new_row]
	end_inds = inds[1, new_row - 1]
	x1 = np.amax(start_inds)
	x2 = np.amin(end_inds)
	iters = math.floor((coords[3] - coords[2])/600)
	#loop through image 500 at a time
	for i in range(iters):
		#start = i*600 - (int(coords[3] - northing))
		start = 50
		arr = dset[:,start:start+500,x1:x2]
		arr = unravel(arr)
		lidar = getLidar(x1,start,map_info,shape)
		createImage(arr,lidar,shape,i)
	return




def getLidar(x,y,map_info,shape):
	path = '/media/zach/AOP-NEON1-4/D17/SJER/2013/SJER_L3/SJER_Lidar/CHM/'
	easting = float(map_info[3])
	northing = float(map_info[4])
	x1 =  int(math.floor((easting + x)/1000)*1000)
	y1 = int(math.ceil((northing - y)/1000)*1000-1000)
	file = path + '2013_SJER_1_' + str(x1) + '_' + str(y1) + '_CHM.tif'
	print(file)
	lidar = gdal.Open(file)
	lidar_array = np.array(lidar.GetRasterBand(1).ReadAsArray())
	column = int(round(easting + x)-x1)
	row = int(y1-round(northing - y))
	lidar_array = lidar_array[row:shape[0]+row,column:shape[1]+column]
	lidar_array[lidar_array == -9999] = 0
	if not np.array_equal(lidar_array[lidar_array == 0], lidar_array):
		lidar_array = featureNorm(lidar_array, True)
	return lidar_array

def createImage(X,lidar,X_shape,counter):	
	img = np.zeros((X_shape[0],X_shape[1],3))
	print(img.shape)
	for i in range(X_shape[1]):
		img[:,i,0] = X[i*X_shape[0]:i*X_shape[0]+X_shape[0],0]
		img[:,i,1] = X[i*X_shape[0]:i*X_shape[0]+X_shape[0],1]
		#img[:,i,2] = X[i*X_shape[0]:i*X_shape[0]+X_shape[0],2]

	img[:,:,2] = lidar
	for i in range(3):
		img[:,:,i] = (255/img[:,:,i].max() * (img[:,:,i] - img[:,:,i].min()))
	fig,ax = plt.subplots(1)
	ax.imshow(img)
	fname = '../processed-data/test/composite_'  + str(counter) + '.jpg'
	scipy.misc.imsave(fname,img)
	return


def unravel(arr):
	img_shape = arr.shape
	X = np.zeros((img_shape[1]*img_shape[2], img_shape[0]))
	for i in range(img_shape[2]):
		X[img_shape[1]*i:img_shape[1]*i+img_shape[1], :] = np.transpose(arr[:,:,i])
	X = featureNorm(X)
	X = reduceDims(X)
	return X


#Normalize data
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

#Fit pca to field data
def reduceDims(X):
	pca = joblib.load('PCA_fit.pkl')
	X = pca.transform(X)
	return X



field_data = '../raw-data/neon_plants.csv'
df = pd.read_csv(field_data, skiprows = 0)
east = df['easting']
north = df['northing']
y, rad, levels = getTarget(df)
field_coords = np.vstack((east, north, rad, y))
plots, labels = getPlots(field_coords[0:2,:])

x1 = int(round(np.amin(plots[0,:]) - 20))
x2 = int(round(np.amax(plots[0,:]) + 20))
y1 = int(round(np.amin(plots[1,:]) - 20))
y2 = int(round(np.amax(plots[1,:]) + 20))
bbox = np.array([x1,x2,y1,y2])

img = np.zeros((y2-y1,x2-x1), dtype=bool)

path = '/media/zach/AOP-NEON1-4/D17/SJER/2013/SJER_L1/SJER_Spectrometer/Reflectance/'
files = os.listdir(path)
files = files[2:-1]
for file in files:
	f = h5py.File(path + file, 'r')
	shape = f['Reflectance'].shape
	map_info = f['/map info'][0].decode().split(',')
	easting = float(map_info[3])
	northing = float(map_info[4])
	coords = np.array([easting, easting+shape[2], northing-shape[1], northing])
	coords = np.round(coords)
	coords = coords.astype(int)
	print(coords)
	limits = np.greater([coords[0], x2, coords[2], y2], [x1, coords[1], y1, coords[3]])
	coords[limits] = bbox[limits]
	check = img[coords[2]:coords[3], coords[0]:coords[1]]
	cleanData(f, coords, check)
	break



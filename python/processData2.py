# Script to processraw data to pass to fast rcnn
import math
import numpy as np 
import h5py
import os
from sklearn import decomposition
from sklearn.externals import joblib
from sklearn.cluster import KMeans
import scipy.misc
import re
from osgeo import gdal
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def getFiles():
	field_data = "C:/temp/neon_plants.csv"
	df = pd.read_csv(field_data, skiprows = 0)
	east = df['easting']
	north = df['northing']
	y, rad = getTarget(df)
	field_coords = np.vstack((east, north, rad, y))
	field_coords = field_coords[:,field_coords[0,:] <= 257000]
	field_coords = field_coords[:,field_coords[0,:] >=255000]
	clusters, labels = getCentroids(field_coords[0:2,:])
	path = 'E:/D17/SJER/2013/SJER_L1/SJER_Spectrometer/Reflectance/'
	files = os.listdir(path)
	counter = 0
	i = 1
	match = False
	for i in range(len(clusters)):
		while not match:
			i = 5
			f = h5py.File(path + files[counter], 'r')
			shape = f['Reflectance'].shape
			print(clusters)
			map_info = f['/map info'][0].decode().split(',')
			easting = float(map_info[3])
			northing = float(map_info[4])
			match = clusters[i,0] >= easting + 120 and clusters[i,0] <= easting + shape[2] - 120 \
				    and clusters[i,1] <= northing and clusters[i,1] >= northing - shape[1]
			counter = counter + 1
			if counter == len(files):
				break
		y_start = int(northing - clusters[i,1] - 20)
		y_end = y_start + 40
		x_start = int(clusters[i,0] - easting - 20)
		x_end = x_start + 40
		box = ((x_start, y_start), (x_end, y_end))
		dset = f['Reflectance']
		img = cleanData(dset,map_info,box)
		coords = ((x_start + easting,x_end + easting),(northing - y_start,northing - y_end))
		getLabels(coords,field_coords[0:3,labels == i], img, i)
	return 

def getTarget(df):
	species = df['taxonid']
	labels, levels = pd.factorize(species)
	y = labels
	rad = df['canopydiam_90deg']
	return y, rad/2

def getCentroids(X):
	kmeans = KMeans(n_clusters = 11).fit(np.transpose(X))
	return (kmeans.cluster_centers_, kmeans.labels_)

#Extract all columns to be used as features in the svm
def getFeatures(df):
	ident = 'nm_'
	names = []
	for column in df:
		if re.match(ident, column):
			names.append(column)
	X = df[names]
	X = X.as_matrix()
	return X

#Cut of edges
def cleanData(dset, map_info, box):
	X = dset[:, box[0][1]:box[1][1], box[0][0]:box[1][0]]
	x = box[0][0]
	y = box[0][1]
	shape = X.shape[1:3]
	X = unravel(X)

	lidar = getLidar(x,y,map_info,shape)
	lidar = featureNorm(lidar,True)
	#if lidar.shape == shape:
	counter = 1
	img = createImage(X,lidar,shape,counter)

	return img


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


def unravel(arr):
	img_shape = arr.shape
	X = np.zeros((img_shape[1]*img_shape[2], img_shape[0]))
	for i in range(img_shape[2]):
		X[img_shape[1]*i:img_shape[1]*i+img_shape[1], :] = np.transpose(arr[:,:,i])
	X[X == 15000] = np.mean(X != 15000)
	X = featureNorm(X)
	X = reduceDims(X)
	return X

def getLidar(x,y,map_info,shape):
	path = '../SJER_Lidar/CHM/'
	easting = float(map_info[3])
	northing = float(map_info[4])
	x1 =  int(math.floor((easting + x)/1000)*1000)
	y1 = int(math.ceil((northing - y)/1000)*1000-1000)
	file = path + '2013_SJER_1_' + str(x1) + '_' + str(y1) + '_CHM.tif'
	lidar = gdal.Open(file)
	lidar_array = np.array(lidar.GetRasterBand(1).ReadAsArray())
	column = round(round(easting + x)-x1)
	row = round(y1-round(northing - y))
	lidar_array = lidar_array[row:shape[0]+row,column:shape[1]+column]
	lidar_array[lidar_array == -9999] = 0
	if not np.array_equal(lidar_array[lidar_array == 0], lidar_array):
		lidar_array = featureNorm(lidar_array, True)

	#check spatial extent of lidar
	gt = np.array(lidar.GetGeoTransform())
	PL = np.array([1, column, -row])
	x = np.dot(gt[0:3], PL)
	y = np.dot(gt[3:6], PL)
	print(file)
	return lidar_array

def getLabels(coords, field_coords, img, counter):
	field_coords[0,:] = field_coords[0,:] - coords[0][0]
	field_coords[1,:] = coords[1][0] - field_coords[1,:]
	print(field_coords)
	fig,ax = plt.subplots(1)
	ax.imshow(img)
	for i in range(field_coords.shape[1]):
		circ = patches.Circle((field_coords[0,i], field_coords[1,i]), field_coords[2,i], linewidth=1, facecolor='none')
		ax.add_patch(circ)
	plt.show()
	#fname = '../processed-data/composite_train_'  + str(counter) + '.jpg'
	#scipy.misc.imsave(fname,img)
	return 






def createImage(X,lidar,X_shape,counter):	
	img = np.zeros((X_shape[0],X_shape[1],3))
	for i in range(X_shape[1]):
		img[:,i,0] = X[i*X_shape[0]:i*X_shape[0]+X_shape[0],0]
		img[:,i,1] = X[i*X_shape[0]:i*X_shape[0]+X_shape[0],1]
		#img[:,i,2] = X[i*X_shape[0]:i*X_shape[0]+X_shape[0],2]

	img[:,:,2] = lidar
	for i in range(3):
		img[:,:,i] = (255/img[:,:,i].max() * (img[:,:,i] - img[:,:,i].min()))
	#fname = '../processed-data/composite_'  + str(counter) + '.jpg'
	#scipy.misc.imsave(fname,img)
	return img







getFiles()
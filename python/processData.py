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
import gdal
import re
import pandas as pd
import matplotlib.pyplot as plt


def getFiles():
	field_data = '../raw-data/neon_plants.csv'
	df = pd.read_csv(field_data, skiprows = 0)
	east = df['easting']
	north = df['northing']
	y = getTarget(df)
	field_coords = np.vstack((east, north, y))
	field_coords = field_coords[:,field_coords[0,:] <= 257000]
	field_coords = field_coords[:,field_coords[0,:] >=255000]
	clusters = getCentroids(field_coords[0:2,:])
	path = '/media/zach/AOP-NEON1-4/D17/SJER/2013/SJER_L1/SJER_Spectrometer/Reflectance/'
	files = os.listdir(path)
	counter = 0
	i = 1
	match = False
	while not match:
		print(counter)
		f = h5py.File(path + files[counter], 'r')
		shape = f['Reflectance'].shape
		map_info = f['/map info'][0].decode().split(',')
		easting = float(map_info[3])
		northing = float(map_info[4])
		match = clusters[i,0] >= easting and clusters[i,0] <= easting + shape[2] \
			    and clusters[i,1] <= northing and clusters[i,1] >= northing - shape[1]
		counter = counter + 1
		if counter == len(files):
			counter = 0
			i = i+1
	y_start = int(northing - clusters[i,1])
	dset = f['Reflectance']
	coords = cleanData(dset,map_info,y_start)
	#getLabels(coords,field_coords)
	return 

def getTarget(df):
	species = df['taxonid']
	labels, levels = pd.factorize(species)
	y = labels
	return y

def getCentroids(X):
	kmeans = KMeans(n_clusters = 11).fit(np.transpose(X))
	return kmeans.cluster_centers_

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
def cleanData(dset, map_info, y_start):
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
	X = dset[:,y_start:y_start + 600,x1:x2]
	x = x1
	y = y_start
	shape = X.shape[1:3]
	#img[img == 15000] = 0
	X = unravel(X)

	lidar = getLidar(x,y,map_info,shape)
	lidar = featureNorm(lidar,True)
	#if lidar.shape == shape:
	counter = 1
	createImage(X,lidar,shape,counter)
	x_coords = (x1+easting,x2+easting)
	y_coords = (northing-y_start, northing-y_start-600)
	return (x_coords,y_coords)


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
	column = int(round(easting + x)-x1)
	row = int(y1-round(northing - y))
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

def getLabels(coords,field_coords):
	for i in range(2):
		for j in range(2):
			field_coords = field_coords[:,field_coords[i,:] >= coords[i][j]]
	field_coords[0,:] = field_coords[0,:] - coords[0][0]
	field_coords[1,:] = field_coords[1,:] - coords[1][0]
	labels = np.zeros((coords[1][0] - coords[1][1], coords[0][1] - coords[0][0]))
	for i in range(field_coords.shape[1]):
		labels[field_coords[1,i],field_coords[0,i]] = field_coords[2,i]
	plt.hist(labels)
	plt.show()
	return






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
	fname = '../processed-data/composite_'  + str(counter) + '.jpg'
	scipy.misc.imsave(fname,img)
	return







getFiles()
#cleanData(dset,map_info)
#(img,x,y) = parseImage(dset, coords, f)



#write to geotiff
#map_info = f["/map info"][0].decode().split(',')
#x_min = float(map_info[3]) + x
#y_max = float(map_info[4]) - y
#cols = img.shape[1]
#rows = img.shape[0]
#driver = gdal.GetDriverByName('GTiff')
#outRaster = driver.Create('test.tif', cols, rows, 1, gdal.GDT_Byte)
#outRaster.SetGeoTransform((x_min, 1, 0, y_max, 0, 1))
#outband = outRaster.GetRasterBand(1)
#outband.WriteArray(img)
#outRasterSRS = osr.SpatialReference()
#outRasterSRS.ImportFromEPSG(4326)
#outRaster.setProjection(outRasterSRS.ExportToWkt())
#outband.FlushCache()


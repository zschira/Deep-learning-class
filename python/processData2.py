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
import xml.etree.cElementTree as ET
#import cv2


def getFiles():
	field_data = "C:/temp/neon_plants.csv"
	df = pd.read_csv(field_data, skiprows = 0)
	east = df['easting']
	north = df['northing']
	y, rad, levels = getTarget(df)
	field_coords = np.vstack((east, north, rad, y))
	field_coords = field_coords[:,field_coords[0,:] <= 257000]
	field_coords = field_coords[:,field_coords[0,:] >=255000]
	plots, labels = getPlots(field_coords[0:2,:])
	path = 'F:/D17/SJER/2013/SJER_L1/SJER_Spectrometer/Reflectance/'
	files = os.listdir(path)
	counter = 0
	match = False
	i=0
	print(plots.shape)
	while i < plots.shape[1]:
		counter = 0
		while not match:
			f = h5py.File(path + files[counter], 'r')
			shape = f['Reflectance'].shape
			map_info = f['/map info'][0].decode().split(',')
			easting = float(map_info[3])
			northing = float(map_info[4])
			match = plots[0,i] >= easting + 120 and plots[0,i] <= easting + shape[2] - 120 \
				    and plots[1,i] <= northing and plots[1,i] >= northing - shape[1]
			counter = counter + 1
			if counter == len(files)-1:
				print(counter,len(files))
				break
		else:
			y_start = int(northing - plots[1,i] - 20)
			y_end = y_start + 40
			x_start = int(plots[0,i] - easting - 20)
			x_end = x_start + 40
			box = ((x_start, y_start), (x_end, y_end))
			dset = f['Reflectance']
			img = cleanData(dset,map_info,box,i)
			if img.size:
				coords = ((x_start + easting,x_end + easting),(northing - y_start,northing - y_end))
				getLabels(coords,field_coords[:,labels == i], img.shape, i, levels)
		i = i+1
	return 

def getTarget(df):
	species = df['taxonid']
	labels, levels = pd.factorize(species)
	y = labels
	rad = df['canopydiam_90deg']
	return y, rad/2, levels

def getPlots(X):
	file = 'C:/temp/plots.csv'
	df = pd.read_csv(file, skiprows = 0)
	east = df['easting']
	north = df['northing']
	plots = np.vstack((east,north))
	labels = np.zeros(len(X[0,:]))
	lab1 = labels
	lab2 = labels
	for i in range(len(plots)):
		temp1 = X[0,:]-east[i]
		temp2 = X[1,:]-north[i]
		lab1[abs(temp1) <= 20] = 1
		lab2[abs(temp2) <= 20] = 1
		labels[lab1 == lab2] = i
	return plots, labels



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
def cleanData(dset, map_info, box, counter):
	X = dset[:, box[0][1]:box[1][1], box[0][0]:box[1][0]]
	x = box[0][0]
	y = box[0][1]
	shape = X.shape[1:3]
	if np.mean(X) == 15000:
		return np.array([])
	X = unravel(X)
	if not X.size:
		return X


	lidar = getLidar(x,y,map_info,shape)
	lidar = featureNorm(lidar,True)
	#if lidar.shape == shape:
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
	if not sigma[0]:
		return np.array([])
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
	if not X.size:
		return np.array([])
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
	return lidar_array

def getLabels(coords, field_coords, shape, counter, labels):
	field_coords[0,:] = field_coords[0,:] - coords[0][0]
	field_coords[1,:] = coords[1][0] - field_coords[1,:]
	#labels = np.zeros((shape[0], shape[1]))
	root = ET.Element("root")
	for i in range(field_coords.shape[1]):
		obj = ET.SubElement(root, 'object')
		bbox = ET.SubElement(obj, 'bndbox')
		rad = field_coords[2,i]
		x1 = math.floor(field_coords[1,i]-rad)
		x2 = math.ceil(field_coords[1,i]+rad)
		y1 = math.floor(field_coords[0,i]-rad)
		y2 = math.ceil(field_coords[0,i]+rad)
		box = np.array([x1,x2,y1,y2])
		box[box < 0] = 0
		box[box > 39] = 39
		ET.SubElement(bbox, 'xmin').text = str(box[0])
		ET.SubElement(bbox, 'xmax').text = str(box[1])
		ET.SubElement(bbox, 'ymin').text = str(box[2])
		ET.SubElement(bbox, 'ymax').text = str(box[3])
		name = labels[field_coords[3,i]]
		ET.SubElement(obj, 'name').text = name
	tree = ET.ElementTree(root)
	tree.write('../processed-data/composite_' + str(counter) + '.xml')
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
	fname = '../processed-data/composite_'  + str(counter) + '.jpg'
	scipy.misc.imsave(fname,img)
	return img







getFiles()
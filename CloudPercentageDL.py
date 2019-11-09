# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:33:06 2019

@author: Rohit Gandikota (NR02440)
"""

from osgeo import gdal, ogr, osr
from keras import *
import os 

from random import shuffle
import numpy as np
import math
from keras.optimizers import *
from keras.layers import *
import matplotlib.pyplot as plt
from keras.regularizers import l2
from sunposition import sunpos 
from DN2TOA import GenerateTOAImages

def CalCloudPer(args):
	az, zen = sunpos(args.t, args.lat, args.lon, 0)[:2]


	if args.n == 4:
		num_classes = 4
	else:
		num_classes = 2

	test_path = args.file
	test_img = gdal.Open(test_path)
	print(np.shape(test_img))
	test_image_TOA = GenerateTOAImages(args.file, 4, 2, args.t, 90-zen)
	print(np.shape(test_image_TOA))
	X_test = np.reshape(test_image_TOA, (4,-1))
	X_test = np.swapaxes(X_test,0,1)
	#X_test = np.expand_dims(X_test,axis=0)
	
	model = Sequential()
	model.add(Dense(16, activation='relu',  input_shape=(4,)))
	model.add(BatchNormalization())
	model.add(Dense(32,  activation='relu'))
	model.add(BatchNormalization())
	model.add(Dense(64,  activation='relu'))
	model.add(BatchNormalization())
	model.add(Dense(32,  activation='relu'))
	model.add(BatchNormalization())
	model.add(Dense(16, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dense(num_classes, activation='softmax'))
	
	if args.n == 4:
		model.load_weights("/maintenance/rkrg_dnd/DLProjects/PixelLULC/BestModel.h5")

	labels_test = model.predict_classes(X_test,batch_size=1000000, verbose=1)
	labels_test = labels_test.reshape((test_img.RasterYSize, test_img.RasterXSize))

	def WriteRaster(InputArray, OutputFile, NROWS, NCOLS, wkt_projection, geotransform):
	    driver = gdal.GetDriverByName("GTiff")
	    dataset = driver.Create(OutputFile, NCOLS, NROWS, 1, gdal.GDT_Byte)
	    dataset.SetGeoTransform(geotransform)
	    dataset.SetProjection(wkt_projection)
	    dataset.GetRasterBand(1).WriteArray(InputArray)
	    dataset.FlushCache()
	    return None

	#WriteRaster(labels_test, 'Cloud_'+args.file, test_img.RasterYSize, test_img.RasterXSize,test_img.GetProjection(),test_img.GetGeoTransform())

	cloud_percentage = (len(labels_test[labels_test==0])/len(labels_test[labels_test==labels_test]))*100
	label = labels_test[:int(np.shape(labels_test)[0]/2),:int(np.shape(labels_test)[1]/2)]
	cp1 = (len(label[label==0])/len(label[label==label]))*100
	label = labels_test[:int(np.shape(labels_test)[0]/2),int(np.shape(labels_test)[1]/2):]
	cp2 = (len(label[label==0])/len(label[label==label]))*100
	label = labels_test[int(np.shape(labels_test)[0]/2):,:int(np.shape(labels_test)[1]/2)]
	cp3 = (len(label[label==0])/len(label[label==label]))*100
	label = labels_test[int(np.shape(labels_test)[0]/2):,int(np.shape(labels_test)[1]/2):]
	cp4 = (len(label[label==0])/len(label[label==label]))*100
	return cp1, cp2, cp3, cp4, cloud_percentage









if __name__ == '__main__':
	from argparse import ArgumentParser
	import datetime, sys 
	parser = ArgumentParser(prog='CloudPercentageDL', description='Computes cloud percentage using Deep Learning models given Raw tiff file')
	parser.add_argument('-f,--file', dest='file', type=str, help = 'Give the full file location for the tiff file to which the cloud percentage needs to be calculated')
	parser.add_argument('-t,--time', dest='t',type=str, help= 'Date and Time (UTC) in "YYYY-MM-DD hh:mm:ss.ssssss" format or (UTC) POSIX timestamp')
	parser.add_argument('-lat,--latitude', dest='lat',type=float, help= 'latitude in decimal degrees, positive for north and negative for south')
	parser.add_argument('-lon,--longitude', dest='lon',type=float, help= 'longitude in decimal degrees, positive for east and negative for west')
	parser.add_argument('-num,--NumberOfClasses', dest='n',type= float, default= 4, help= 'Number of classes to be classified (2 or 4). Deafult set to "2" either cloud or not')
	

	args = parser.parse_args()

	c1,c2,c3,c4,c = CalCloudPer(args)
	print('NW: ' + str(c1) + ' NE: ' + str(c2) + ' SW: ' + str(c3) + ' SE: ' + str(c4) + ' Total: ' + str(c) )  

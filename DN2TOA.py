# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:33:06 2019

@author: Rohit Gandikota (NR02440)
"""

import gdal
import sys
import os
import numpy as np
import datetime
import bisect
import math

Earth_Sun_Distance_Table=[

[1,15,32,46,60,74,91,106,121,135,152,166,182,196,213,227,242,258,274,288,305,319,335,349,365],
[.9832,.9836,.9853,.9878,.9909,.9945,.9993,1.0033,1.0076,1.0109,1.0140,1.0158,1.0167,1.0165,1.0149,1.0128,1.0092,1.0057,1.0011,.9972,.9925,.9892,.9860,.9843,.9833]

]

L3_ExoAtmospheric_Irradiance=[xxx,xxx.92,xxx.96,xxx.38]
C3_ExoAtmospheric_Irradiance=[xxx.78, xxx.01, xxx, xxx.86]
C2E_ExoAtmospheric_Irradiance=[xxx.91, xxx.37, xxx.38, xxx.22]

#--- FUNCTIONS ----

def ParseMeta(text_file):
    with open(text_file) as f:
        content = f.readlines()
    return content
def GetEarthSunDistance(time,Earth_Sun_Distance_Table):
    date=time[0:10]
    fdate=datetime.datetime.strptime(date,'%Y-%m-%d')
    dayofyear=fdate.timetuple().tm_yday
    #print(dayofyear)
    right_index=bisect.bisect_right(Earth_Sun_Distance_Table[0], dayofyear)
    left_index=right_index-1
    x1=Earth_Sun_Distance_Table[0][left_index]
    x2=Earth_Sun_Distance_Table[0][right_index]
    y1=Earth_Sun_Distance_Table[1][left_index]
    y2=Earth_Sun_Distance_Table[1][right_index]
    x=dayofyear
    y=((y2-y1)/(x2-x1))*(x-x1)+y1
    return (y)

def GenerateTOAImages(bands, No_Bands,StartOffset, SceneCenterTime, SunElevationAtCenter, L3_ExoAtmospheric_Irradiance):

    #----------------- FIND ALL INPUT RASTERSIN INPUT LOCATION --------------
    final_bands = []
    index=0
    for band in bands:
        #print(np.shape(band))
        E0=L3_ExoAtmospheric_Irradiance[index]
        image_ar= band
        lmin= 0.0
        lmax= 26.12
        qmin=0
        qmax=2**(int(11))-1
        slope=(lmax-lmin)/(qmax-qmin)
        ##yin=lmin
        new_ar=slope*(image_ar-qmin)+lmin  #RADIANCE Array
        #Convert to TOA
        es_distance=GetEarthSunDistance(SceneCenterTime,Earth_Sun_Distance_Table)
        sunzenith=math.radians(90-float(SunElevationAtCenter))
        new_ar=(math.pi*(es_distance**2)/E0*math.cos(sunzenith))*new_ar
        # #print(new_ar)
        # driver = image_ds.GetDriver()
        # #print driver
        # base_name=os.path.basename(im_file)
        # outDs = driver.Create(orasdir+"/"+base_name, Width,Height, 1,gdal.GDT_Float32)
        # outBand = outDs.GetRasterBand(1)
        # outBand.WriteArray(new_ar)

        # outBand.SetNoDataValue(-99)

        # # georeference the image and set the projection
        # outDs.SetGeoTransform(image_ds.GetGeoTransform())
        # outDs.SetProjection(image_ds.GetProjection())
        # outBand.FlushCache()
        index += 1
         #print(np.shape(new_ar))
        final_bands.append(new_ar)
    #print(np.shape(final_bands))
    return np.array(final_bands)
#----------

# #USAGE
# #python DN2TOA.py inputrasdir irasformat outputrasdir
# irasdir=sys.argv[1]
# #irasformat=sys.argv[2]
# orasdir=sys.argv[2]

# GenerateTOAImages(irasdir,orasdir,4,2,L3_ExoAtmospheric_Irradiance)






    #print(image_ar)

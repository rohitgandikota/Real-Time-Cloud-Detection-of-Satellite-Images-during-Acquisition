# -*- coding: utf-8 -*
"""
Created on Tue Oct 22 15:33:06 2019

@author: Rohit Gandikota (NR02440)
"""

import os
import numpy as np
import gdal
import matplotlib.pyplot as plt
import time
from sunposition import sunpos
from DN2TOA import GenerateTOAImages
from keras import *
import os 
import datetime
from random import shuffle
import numpy as np
import math
from keras.optimizers import *
from keras.layers import *
import matplotlib.pyplot as plt
from keras.regularizers import l2
C3_ExoAtmospheric_Irradiance=[xxx.xx, yyy.yy, www.ww, ddd.dd]
def ReadDecomp(F1, F2, F3, F4,SAT_ID,SEN_ID):
    if(SAT_ID=="C03" and SEN_ID=="MX"):
        t = "2019-11-06 10:22:56"
        lat = 21.45
        lon = 78.12
        cloud = 0
        total = 0 
        ##################################### Deep Learning Code #########################################
      
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
        model.add(Dense(2, activation='softmax'))
        model.load_weights("/maintenance/rkrg_dnd/DLProjects/RealTimeCloudC3/Model_Final.h5")

        ##################################################################################################        
        files = [F1, F2, F3, F4]
        AUX_LEN=200
        CCD_COUNT=16080
        NUMBER_OF_BITS=11
        BYTE_COUNT=2
        LINE_BYTES=CCD_COUNT*BYTE_COUNT
        while not (os.path.exists(F1) and os.path.exists(F2) and os.path.exists(F3) and os.path.exists(F4)):
            time.sleep(1)
            print('waiting for the pass')
        start_point = 0
        file1_size = 0
        
        while True:
            f_sizes = []
            for i in range(1,5):
                f2 = os.stat(files[i-1])
                file2_size = f2.st_size
                comp = file2_size - file1_size
                f_sizes.append(comp)
            f_sizes = np.array(f_sizes)
            min_lines = np.min((f_sizes//(LINE_BYTES+AUX_LEN)))
            #print(file1_size, file2_size, comp
            
            if min_lines > 0:
                status = 0 
                print('Reading newly received data')
                # FILE_SIZE= os.path.getsize(FILE_NAME)
                # print(FILE_SIZE)
                # if(not NUM_LINES.is_integer()):
                #     print(' THE DATA MIGHT BE CORRUPTED OR IS INCOMPLETE, IGNORING THE INCOMPLETE LINE AND READING COMPLETE LINES')



                #- READING BINARY ---
                # 0 (or os.SEEK_SET): means your reference point is the beginning of the file
                # 1 (or os.SEEK_CUR): means your reference point is the current file position
                # 2 (or os.SEEK_END): means your reference point is the end of the file
                bands = []
                for i in range(1,5):
                    file= open(files[i-1],"rb")
                    data= []
                    file.seek(start_point)
                    end_point = start_point + min_lines * (LINE_BYTES+AUX_LEN)
                    for line in range(min_lines):

                        # - EACH LINE CONSISTS OF BYTE_COUNT*CCD_COUNT BYTES
                        # - DETERMINE THE NUMBER OF LINES = FILE_SIZE(bytes)/12000
                        # - IF NOT EXACTLY DIVISIBLE TAKE FLOOR. ( RETURN A WARNING IF POSSIBLE )
                        file.seek(AUX_LEN,1)
                        x=np.fromfile(file,np.uint16,CCD_COUNT)
                        #print(x)
                        data.append(x)
                       # break
                        #print(np.shape(np.array(data)), min_line
                    bands.append(np.array(data))
                        #print(line)
                start_point = end_point
                file1_size = file1_size + min_lines * (LINE_BYTES+AUX_LEN)
                print(np.shape(np.array(bands)),min_lines)
                az, zen = sunpos(t,lat,lon,0)[:2]
                print('Generating TOA Images')
                bands = GenerateTOAImages(np.array(bands), 4,1, t, 90-zen, C3_ExoAtmospheric_Irradiance)
                print(np.shape(bands))
                print('TOA Images are generated and are being classified to check for cloud pixels')
                bands = np.array(bands)
                X_test = np.reshape(bands, (4,-1))
                X_test = np.swapaxes(X_test,0,1)
                print(np.shape(X_test))
                labels_test = model.predict_classes(X_test,batch_size=10000000, verbose=1)
                total = total + len(labels_test)
                labels_test = labels_test.reshape(np.shape(bands)[1], np.shape(bands)[2])
                print('Classification Done using Deep Neural Network')
                cloud = cloud + np.count_nonzero(labels_test==0)
                 
                #return(labels_test, min_lines)
            else:
                if status == 0:
                    t1 = datetime.datetime.now()
                    status = 1
                else:
                    diff = datetime.datetime.now() - t1
                    if diff.seconds > 15:
                        
                        print('Cloud Percentage: '+ str((cloud/total)*100))
                        return 0
                 
                pass
    else:
        return None





def ReadDecompFullwithLUT(FILE_NAME,SAT_ID,SEN_ID,LUT_PATH):
    lut=ReadLut(LUT_PATH,SAT_ID,SEN_ID)
    if(SAT_ID=="RS2A" and SEN_ID=="AWIFS"):
        AUX_LEN=100
        CCD_COUNT=6000
        NUMBER_OF_BITS=12
        BYTE_COUNT=2
        LINE_BYTES=CCD_COUNT*BYTE_COUNT

        FILE_SIZE= os.path.getsize(FILE_NAME)
        print(FILE_SIZE)
        NUM_LINES=(FILE_SIZE)/(LINE_BYTES+100)
        print(NUM_LINES)

        if(not NUM_LINES.is_integer()):
            print(' THE DATA MIGHT BE CORRUPTED OR IS INCOMPLETE, IGNORING THE INCOMPLETE LINE AND READING COMPLETE LINES')



        #- READING BINARY ---
        # 0 (or os.SEEK_SET): means your reference point is the beginning of the file
        # 1 (or os.SEEK_CUR): means your reference point is the current file position
        # 2 (or os.SEEK_END): means your reference point is the end of the file

        file=open(FILE_NAME,"rb")
        data=[]
        #file.seek((100+LINE_BYTES)*START_LINE)
        for line in range(0,int(NUM_LINES)):

            # - EACH LINE CONSISTS OF BYTE_COUNT*CCD_COUNT BYTES
            # - DETERMINE THE NUMBER OF LINES = FILE_SIZE(bytes)/12000
            # - IF NOT EXACTLY DIVISIBLE TAKE FLOOR. ( RETURN A WARNING IF POSSIBLE )
            file.seek(100,1)
            x=np.fromfile(file,np.uint16,6000)
            #print(x)
            data.append(x)
        data=np.array(data)
        data_lut=[]
        for line in range(0,int(NUM_LINES)):
            if(line%1000==0):
                print('LINE '+str(line)+' OF TOTAL '+str(NUM_LINES)+' Lines')
            x=[]
            for pix in range(0,CCD_COUNT):
                x.append(lut[pix][data[line][pix]])
            x=np.array(x)
            data_lut.append(x)
        del data,lut
        data_lut=np.array(data_lut)

        return(data_lut,NUM_LINES)
    else:
        return None


def WriteDecomp(data,OUTPUT_FILE_NAME,ncols,nrows):
    print('WRITING THE IMAGE')
    gdal.AllRegister()
    driver = gdal.GetDriverByName('GTiff')#.Create('myraster.tif',ncols, nrows, 1 ,gdal.GDT_UInt16)
    outDs = driver.Create(OUTPUT_FILE_NAME, ncols, nrows,1, gdal.GDT_UInt16)
    if outDs is None:
        print ('Could not create '+str(OUTPUT_FILE_NAME) )
        sys.exit(1)

    outDs.GetRasterBand(1).WriteArray(data)
    outDs.FlushCache()
#outData = numpy.zeros((rows,cols), numpy.int16)

def ReadLut(LUT_PATH,SAT_ID,SEN_ID):
    if(SAT_ID=="RS2A" and SEN_ID=="AWIFS"):
        #AUX_LEN=100

        CCD_COUNT=6000
        NUMBER_OF_BITS=12
        BYTE_COUNT=2
        DYN_RANGE=2**NUMBER_OF_BITS
        #LUT FORMAT ==
        #---0---||----1----||----2----||-- detectors
        #4096 , 2bytes values for each detector
        file=open(LUT_PATH,"rb")
        lut=[]
        for det in range(0,CCD_COUNT):
            x=np.fromfile(file,np.uint16,DYN_RANGE)
            lut.append(x)
        #print(lut[500][67],lut[3500][67])
        #plt.plot(lut[5999])
        #plt.show()
        #print(len(lut))
        return lut
def ReadDecompFull(FILE_NAME,SAT_ID,SEN_ID):
    #lut=ReadLut(LUT_PATH,SAT_ID,SEN_ID)
    if(SAT_ID=="RS2A" and SEN_ID=="AWIFS"):
        AUX_LEN=100
        CCD_COUNT=6000
        NUMBER_OF_BITS=12
        BYTE_COUNT=2
        LINE_BYTES=CCD_COUNT*BYTE_COUNT

        FILE_SIZE= os.path.getsize(FILE_NAME)
        print(FILE_SIZE)
        NUM_LINES=(FILE_SIZE)/(LINE_BYTES+100)
        print(NUM_LINES)

        if(not NUM_LINES.is_integer()):
            print(' THE DATA MIGHT BE CORRUPTED OR IS INCOMPLETE, IGNORING THE INCOMPLETE LINE AND READING COMPLETE LINES')



        #- READING BINARY ---
        # 0 (or os.SEEK_SET): means your reference point is the beginning of the file
        # 1 (or os.SEEK_CUR): means your reference point is the current file position
        # 2 (or os.SEEK_END): means your reference point is the end of the file

        file=open(FILE_NAME,"rb")
        data=[]
        #file.seek((100+LINE_BYTES)*START_LINE)
        for line in range(0,int(NUM_LINES)):

            # - EACH LINE CONSISTS OF BYTE_COUNT*CCD_COUNT BYTES
            # - DETERMINE THE NUMBER OF LINES = FILE_SIZE(bytes)/12000
            # - IF NOT EXACTLY DIVISIBLE TAKE FLOOR. ( RETURN A WARNING IF POSSIBLE )
            file.seek(100,1)
            x=np.fromfile(file,np.uint16,6000)
            #print(x)
            data.append(x)
        data=np.array(data)
        return(data,NUM_LINES)
    else:
        return None







#---------------- GENERIC PUSHBROOM : SPECIFIC AWIFS READER --------------------------
if(__name__=="__main__"):




    F1 = '/maintenance/rkrg_dnd/DLProjects/RealTimeCloudC3/Data/vddmxb1f_f_SAN_c2e.08aug2019'
    F2 = '/maintenance/rkrg_dnd/DLProjects/RealTimeCloudC3/Data/vddmxb2f_f_SAN_c2e.08aug2019'
    F3 = '/maintenance/rkrg_dnd/DLProjects/RealTimeCloudC3/Data/vddmxb3f_f_SAN_c2e.08aug2019'
    F4 = '/maintenance/rkrg_dnd/DLProjects/RealTimeCloudC3/Data/vddmxb4f_f_SAN_c2e.08aug2019'
    data = ReadDecomp(F1,F2,F3,F4,'C03','MX')
    print('Finish')
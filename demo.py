import cv2
import h5py
import numpy as np
from NPDScan import NPDScan
from nms import nms
import sys

minFace = 40
maxFace = 4000
overlap = 0.6
imgFile = sys.argv[1]
modelFile = 'model_frontal.mat'

f = h5py.File(modelFile,'r')
# x=f.get('npdModel')
# for n,v in x.items():
#     print n,np.array(v)
npdModel = {n:np.array(v) for n,v in f.get('npdModel').items()}
img = cv2.imread(imgFile)
I = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rects = NPDScan(npdModel, I, minFace, maxFace)
# rects = [[10,10,20,20]]
numFaces = len(rects)
print('%s faces detected.\n' % numFaces)
rects = nms(rects,overlap)
if numFaces > 0:
    for rect in rects:
        cv2.rectangle(img, (rect[0], rect[1]), (rect[0]+rect[3], rect[1]+rect[4]), (0,255,0), 2)
cv2.namedWindow('img')
cv2.imshow('img',img)
cv2.waitKey(0)

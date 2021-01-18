# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 13:11:01 2020

@author: mnimisha
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from sympy import *


def BestMatch(descriptors1, descriptors2, keypoints1, keypoints2):
    dist_ratio=0.75
    coresp1=[]
    coresp2=[]
    for i in range(descriptors1.shape[0]):
        innerprod=np.inner(np.expand_dims(descriptors1[i,:], axis=0),descriptors2)
        #angles=[math.acos(i) for i in innerprod[0,:]]
        Sortedangles=sorted(innerprod[0,:], reverse=True)
        if (Sortedangles[1]/Sortedangles[0] < dist_ratio):
            idx=np.where(innerprod[0,:]==Sortedangles[0])
            coresp1.append(keypoints1[i].pt)
            coresp2.append(keypoints2[idx[0][0]].pt)
        
    return coresp1, coresp2

def RANSAC(corresp1,corresp2, count):
    TP=0
    while(TP<0.8*(count-4)):
        index=np.random.choice(count,4)
        point1=list(np.array(corresp1)[index])
        point2=list(np.array(corresp2)[index])        
        A=[]
        for i in range(len(point1)):
            A.append([-point2[i][0], -point2[i][1], -1, 0 ,0 ,0, point1[i][0]*point2[i][0], point1[i][0]*point2[i][1], point1[i][0]])
            A.append([0, 0, 0, -point2[i][0], -point2[i][1], -1, point2[i][0]*point1[i][1], point2[i][1]*point1[i][1], point1[i][1]])
       
        hm=Matrix(A).nullspace()
        h=np.reshape(hm[0],(3,3))
        eps=10
        TP=0      
        for i in range(count):
            Cord_new=np.dot(h,np.transpose([corresp2[i][0], corresp2[i][1], 1]))
            Corresp_N=Cord_new[0:2]/Cord_new[2]   
            if math.sqrt(np.sum((Corresp_N-corresp1[i])**2))<eps:
                TP+=1        
    return h



Img1= cv2.imread('IMG1.pgm',0)
Img2= cv2.imread('IMG2.pgm',0)


sift1=cv2.xfeatures2d.SIFT_create()
keypoints1, descriptors1= sift1.detectAndCompute(Img1,None) 


sift2=cv2.xfeatures2d.SIFT_create()
keypoints2, descriptors2= sift2.detectAndCompute(Img2,None) 

[points1,points2]=BestMatch(np.array(descriptors1), np.array(descriptors2), keypoints1, keypoints2)
##RANSAC with the correspondances
Homography=RANSAC(points1,points2, len(points1))


bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
matches = bf.match(descriptors1,descriptors2)
matches = sorted(matches, key = lambda x:x.distance)
#print(np.array(matches).shape)
'''img3 = cv2.drawMatches(Img1, keypoints1, Img2, keypoints2, matches[:50], Img2, flags=2)
plt.imshow(img3)
plt.show()'''

# Extract location of good matches
points1 = np.zeros((len(matches), 2), dtype=np.float32)
points2 = np.zeros((len(matches), 2), dtype=np.float32)
for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt
h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
# Use homography

Homography=Homography.astype(np.float64)
print(h)
print(Homography)
height, width = Img1.shape
Im1Reg = cv2.warpPerspective(Img2, h, (width, height))


plt.imshow(abs(Im1Reg-Img1)/255,cmap='gray', vmin=0, vmax=1)
plt.show()

plt.imshow(cv2.blur(Img1,(3,3)),cmap='gray')
plt.show()

plt.imshow(Im1Reg,cmap='gray')
plt.show()

#Img1=cv2.drawKeypoints(Img1,keypoints1,None)
#cv2.imshow('Input image1', Img1)
#cv2.waitKey(1)
#
#print(Img2.shape)
#Img2=cv2.drawKeypoints(Img2,keypoints2,None)
#cv2.imshow('Input image2', Img2)
#cv2.waitKey(0)


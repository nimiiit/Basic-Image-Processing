# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 20:48:43 2020

@author: mnimisha
"""

''' Given two images with diffreence and the point correspondances, find the change'''
import cv2
import numpy as np
import matplotlib.pyplot as plt


Img1= cv2.imread("IMG1.pgm",0)
Img2= cv2.imread("IMG2.pgm",0)

#Given the point correspondances between them. i.e (x,y) in img1 maps to (x',y') in img2
#We will see in the next assignment how to find these correspondances using Feature extraction
##[x' y']=R[x y]+T R:rotation and T: Translation
A=np.array([[94,249,1,0], [249,-94, 0,1 ],[329, 400, 1,0] ,[400, -329 ,0, 1]])
b=np.array([30,125,158,373])
A_inv=np.linalg.inv(A) 
A_inv_b=A_inv@b

R=[[A_inv_b[0], A_inv_b[1]],[- A_inv_b[1], A_inv_b[0]]]
R_inv=np.linalg.inv(R)   
T=[A_inv_b[2],A_inv_b[3]]
##Now create a target image with all zeros the same size as input
imgH, imgW=Img2.shape
Targ_Img=np.zeros([imgH,imgW])
#print(R_inv)
for x_Targ in range(imgH):
   for y_Targ in range(imgW):
            #Map every target point to the source image by reverse transform R^-1(SX-T) wrt image center
        Src_points=np.dot(R_inv,[[x_Targ-T[0]],[y_Targ-T[1]]])
        x_Src=Src_points[0]
        x_Src_flr=int(np.floor(x_Src))
        y_Src=Src_points[1]
        y_Src_flr=int(np.floor(y_Src))   
        #check if the mapped point is in the source index range or not
        if (x_Src_flr>=0 and y_Src_flr>=0 and x_Src_flr<imgH-1 and y_Src_flr<imgW-1):
            a=x_Src-x_Src_flr
            b=y_Src-y_Src_flr
            #bilinear interpolation
            Targ_Img[x_Targ][y_Targ]=Img2[x_Src_flr][y_Src_flr]*(1-a)*(1-b)+Img2[x_Src_flr][y_Src_flr+1]*(1-a)*(b)+Img2[x_Src_flr+1][y_Src_flr]*(a)*(1-b)+Img2[x_Src_flr+1][y_Src_flr+1]*(a)*(b)
        else:
            Targ_Img[x_Targ][y_Targ]=0
            

         
#
crop_Targ=Targ_Img[0:Img1.shape[0],0:Img1.shape[1]]
Difference=abs(Img1-crop_Targ)/255

fig=plt.figure(figsize=(2, 2))

columns = 2
rows = 2
fig.add_subplot(columns, rows, 1)
plt.imshow(Img1)
plt.title('Image1')
plt.axis('off')
fig.add_subplot(columns, rows, 2)
plt.imshow(Img2)
plt.title('Image2 with difference and warping')
plt.axis('off')
fig.add_subplot(columns, rows, 3)
plt.imshow(Targ_Img/255)
plt.title('Image2 rewarped')
plt.axis('off')
fig.add_subplot(columns, rows, 4)
plt.imshow(Difference)
plt.title('Detected Changes')
plt.axis('off')
plt.show()
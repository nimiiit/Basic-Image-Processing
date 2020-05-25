''' Read an image and translate, rotate and scale 
Use target to source mapping
And bilinear interpolation'''

#Steps
    #Read the image
    #for translation provide tx, ty
    #for rotation provide theta
    #for scaling provide alpha (scale factor)
    #write function for target to source mapping and bilinear interolation

import cv2
import numpy as np
import math


def TargSrcMap_bilinear(*args,**kwargs):
    Src_Img=args[0] 
    #extract the translation, rotation and scale values and set to default if not there
    theta=kwargs.get('theta',0.0)
    scale=kwargs.get('scale',1.0)
    tx=kwargs.get('tx',0.0)
    ty=kwargs.get('ty',0.0)
    #create the matrix of transformations 
    R=[[math.cos(math.pi*theta/180), math.sin(math.pi*theta/180)],[- math.sin(math.pi*theta/180), math.cos(math.pi*theta/180)]]
    R_inv=np.linalg.inv(R)   
    S=np.array([[scale, 0],[0, scale]])
    R_inv_S=np.dot(S,R_inv)
    T=[tx,ty]
    #Now create a target image with all zeros the same size as input
    imgH, imgW=Src_Img.shape
    Targ_Img=np.zeros([imgH,imgW])

    for x_Targ in range(imgH):
        for y_Targ in range(imgW):
            #Map every target point to the source image by reverse transform R^-1(SX-T) wrt image center
            Src_points=np.dot(R_inv_S,[[x_Targ-imgH/2-T[0]],[y_Targ-imgW/2-T[1]]])
            x_Src=Src_points[0]+imgH/2
            x_Src_flr=int(np.floor(x_Src))
            y_Src=Src_points[1]+imgW/2
            y_Src_flr=int(np.floor(y_Src))   
            #check if the mapped point is in the source index range or not
            if (x_Src_flr>=0 and y_Src_flr>=0 and x_Src_flr<imgH-1 and y_Src_flr<imgW-1):
                a=x_Src-x_Src_flr
                b=y_Src-y_Src_flr
                #bilinear interpolation
                Targ_Img[x_Targ][y_Targ]=Src_Img[x_Src_flr][y_Src_flr]*(1-a)*(1-b)+Src_Img[x_Src_flr][y_Src_flr+1]*(1-a)*(b)+Src_Img[x_Src_flr+1][y_Src_flr]*(a)*(1-b)+Src_Img[x_Src_flr+1][y_Src_flr+1]*(a)*(b)
            else:
                Targ_Img[x_Targ][y_Targ]=0
    return Targ_Img    


def Main():
    Src_Img= cv2.imread('lena_translate.pgm',0)
    cv2.imshow('Input image', Src_Img)
    cv2.waitKey(0)
    Targ_Img=TargSrcMap_bilinear(Src_Img, theta=90, scale=0.6)
    cv2.imshow('Target image', Targ_Img/255)
    cv2.waitKey(0)

if __name__=="__main__":
    Main()

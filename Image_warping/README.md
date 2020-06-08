# Image Warping:

We cover basic image transformations like
* Rotation (w.r.t center)
* x-y translations
* Zoom in and out (w.r.t center)

You can always use the existing libraries (Opencv cv2.warpAffine) to perform these operations. But to get a sense of what is happening when we call these functions will be covered here.
The concept to Target to Source Mapping and bilinear interpolation technique to perform these transformations.


**Target to Source Mapping explained:**
   Given a transformation from source to target (say translation or rotation) it is always the practise to do target to source mapping to avoid empty spaces in the warped image.
   
   **How is it done?**   
 - Define a target of same size as image.         
  - Take the index from target and inverse warp these points and see where they fall in the source image.    
  - If the mapped index falls on integer location, we can directly plug in the value to source image to the Target image.   
  - If the mapped index falls on a fraction (where pixel values are not defined) in the source, we use an interpolation method to get the value.    
  - Here we use bilinear interpolation.  
    
   **Target to source mapping with bilinear interpolation explained with diagram**
      <p align="center">
      <img src="https://github.com/nimiiit/Basic-Image-Processing/blob/master/Image_warping/TargetToSourceMap.png" alt="Target to Source Mapping"  width="500" height="250">
      </p>  
      
**Results**
   
 Input Image :  
    <img width="131" alt="Capture1" src="https://user-images.githubusercontent.com/9528369/82805310-6df64200-9ea1-11ea-96b4-c006f3379efe.PNG"> 
    
    
 Transformed Image (After rotation and scaling): 
 
   <img width="133" alt="Capture" src="https://user-images.githubusercontent.com/9528369/82805313-6f276f00-9ea1-11ea-9544-edbe7abee311.PNG">


     

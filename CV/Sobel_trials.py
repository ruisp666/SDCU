# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 11:58:04 2018

@author: sapereira
Part of the self-driving car ND
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y''
    if orient=='x':
        sobel=cv2.Sobel(gray,cv2.CV_64F,1,0)
    else:
        sobel=cv2.Sobel(gray,cv2.CV_64F,0,1)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel=np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel=np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel>thresh_min) & (scaled_sobel<thresh_max)]=1
    # 6) Return this mask as your binary_output image
    #binary_output = np.copy(img) # Remove this line
    return binary_output


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobel_x=cv2.Sobel(gray, cv2.CV_64F,1,0,ksize=sobel_kernel)
    sobel_y=cv2.Sobel(gray, cv2.CV_64F,0,1,ksize=sobel_kernel)
    # 3) Calculate the magnitude 
    #sobel_mag=np.array([np.sqrt(i**2+j**2) for i,j in zip(sobel_x,sobel_y)])
    sobel_mag=np.sqrt(sobel_x**2+ sobel_y**2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel_mag=np.uint8(255*sobel_mag/np.max(sobel_mag))
    # 5) Create a binary mask where mag thresholds are met
    binary_output=np.zeros_like(scaled_sobel_mag)
    binary_output[(scaled_sobel_mag>mag_thresh[0]) & (scaled_sobel_mag<mag_thresh[1])]=1
    # 6) Return this mask as your binary_output image
    #binary_output = np.copy(img) # Remove this line
    return binary_output

def dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobel_x=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=sobel_kernel)
    sobel_y=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx=np.absolute(sobel_x)
    abs_sobely=np.absolute(sobel_y)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    grad_dir=np.arctan2(abs_sobely,abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    binary_output=np.zeros_like(grad_dir)
    binary_output[(grad_dir>=thresh[0]) & (grad_dir<=thresh[1])]=1
    # 6) Return this mask as your binary_output image
    #binary_output = np.copy(img) # Remove this line
    return binary_output


img=mpimg.imread('signs_vehicles_xygrad.png')
ksize=7
img_abs=abs_sobel_thresh(img,ksize, 20,60)
img_mag=mag_thresh(img,ksize,(20,60))
img_dir=dir_thresh(img,ksize,(1.2,1.8))
img_combined=np.zeros_like(img_abs)
img_combined[(img_abs==1 ) & (img_dir==1) | (img_dir==1 ) & (img_mag==1) ]=1
f,(ax1,ax2) = plt.subplots(1,2,figsize=(8,3))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=10)
ax2.imshow(img_combined, cmap='gray')
ax2.set_title('Thresholded Gradient', fontsize=10)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)



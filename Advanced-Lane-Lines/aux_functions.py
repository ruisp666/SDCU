# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 11:23:04 2018

@author: sapereira
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def pipeline_grad_color_thresh(img, s_thresh=(120, 255), sx_thresh=(20, 140)):
    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    #h_channel=hls[:,:,0]
    #l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
        
    # Sobel x
    sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    return color_binary

def corners_unwarp(img, nx, ny, mtx, dist):
    # Pass in your image into this function
    # Write code to do the following steps
    # 1) Undistort using mtx and dist
    undist=cv2.undistort(img,mtx,dist,None,mtx)
    
    # 2) Convert to grayscale
    gray=cv2.cvtColor(undist,cv2.COLOR_BGR2GRAY)
    # 3) Find the chessboard corners
    ret, corners=cv2.findChessboardCorners(gray,(nx,ny),None)
    #
    #warped = np.copy(img) 
    if ret==True:
        drawn_img=cv2.drawChessboardCorners(undist,(nx,ny),corners,ret)
        #drawn_img=cv2.cvtColor(drawn_img,cv2.COLOR_BGR2GRAY)

        src= np.float32([corners[0],corners[1],corners[8],corners[9]])
        #print(src)
        dst=np.float32([[75,75],[250,75],[75,250],[250,250]])
        M=cv2.getPerspectiveTransform(src,dst)
        warped=cv2.warpPerspective(drawn_img,M,gray.shape[::-1])
       
    return warped, M


def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set the number of windows
    nwindows=9
    # Set the width of the windows +/- margin
    margin = 50
    # Set minimum number of pixels found to recenter window
    minpix = 100

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    #print(binary_warped.shape)
    nonzeroy = np.array(nonzero[0])
    #print(np.array(nonzero).shape)
    nonzerox = np.array(nonzero[1])
    #print(nonzerox.shape)
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low =   leftx_current-margin # Update this
        win_xleft_high = leftx_current+margin # Update this
        win_xright_low =rightx_current-margin
        win_xright_high = rightx_current+margin # Update this
        
        # We do not draw the windows on the visualization image, but keep the code for reference
        #cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        #(win_xleft_high,win_y_high),(255,255,0), 2) 
        #cv2.rectangle(out_img,(win_xright_low,win_y_low),
        # (win_xright_high,win_y_high),(255,255,0), 2) 
        
        ###Identify the nonzero pixels in x and y within the window ###
        good_left_inds = binary_warped[win_y_low:win_y_high,win_xleft_low:win_xleft_high].nonzero()
        good_right_inds = binary_warped[win_y_low:win_y_high,win_xright_low:win_xright_high].nonzero()
    
        good_left_inds[0][:]+=win_y_low
        good_left_inds[1][:]+=win_xleft_low
        left_lane_inds.append(good_left_inds)
        good_right_inds[0][:]+=win_y_low
        good_right_inds[1][:]+=win_xright_low
        
        right_lane_inds.append(good_right_inds)
        ### TO-DO: If you found > minpix pixels, recenter next window ###
 
        if len(good_left_inds[0])> minpix:
            leftx_current=int(good_left_inds[1].mean())
        if len(good_right_inds[0]) > minpix:
            rightx_current=int(good_right_inds[1].mean())
    left_lane_inds=np.concatenate(left_lane_inds,axis=1)
    right_lane_inds = np.concatenate(right_lane_inds,axis=1)
    leftx = left_lane_inds[1]
    lefty=left_lane_inds[0]
   
    rightx = np.array(right_lane_inds[1])
    righty= np.array(right_lane_inds[0])
   
    return leftx, lefty, rightx, righty, out_img



def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit` ###
    
    left_fit = np.polyfit(lefty,leftx,2)
    right_fit = np.polyfit(righty,rightx,2)
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    plt.plot(left_fitx,ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')

    return left_fit, right_fit, out_img


def measure_curvature_distance_real(binary_warped,left_fit, right_fit,x_l,y_l):
    '''
    Calculates the curvature  of the road and the vehicle distance from the center in meters.
    '''
    # Define conversions in x and y from pixels space to meters using the measures defined in the notebook.
    ym_per_pix = 30/y_l # meters per pixel in y dimension
    xm_per_pix = 3.7/x_l # meters per pixel in x dimension

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    y_eval = np.max(ploty)*ym_per_pix
    
    left_curverad = (1+(2*left_fit[0]*y_eval+left_fit[1])**2)**(3/2)/(2*np.abs(left_fit[0]))
    
    right_curverad = (1+(2*right_fit[0]*y_eval+right_fit[1])**2)**(3/2)/(2*np.abs(right_fit[0])) 
    
    # Determine the center of the image
    center_x=binary_warped.shape[1]//2
    center_y=binary_warped.shape[0]//2
    # The height of the points is the same, the center of the lane is at the level of the center of the image, and is given as the midpoint between the left and right polynomial fit given at y=binary_warped.shape[0]//2
    left_midx = left_fit[0]*center_y**2 + left_fit[1]*center_y + left_fit[2]
    right_midx = right_fit[0]*center_y**2 + right_fit[1]*center_y + right_fit[2]
    mid_lane_x=(left_midx+right_midx)//2
    cv2.line(binary_warped,(np.int(center_x),np.int(center_y)),(np.int(mid_lane_x),np.int(center_y)),(255,255,0),2)
    distance=np.abs(mid_lane_x - midpoint_x) * xm_per_pix
    return left_curverad, right_curverad, distance, binary_warped






## Project 2: Advanced Lane Finding

In this notebook, we explain the rationale behind the steps in the accompanying notebook adv_lane_finding_notebook'. For clarity, we organize it with respect to the notebook content, and include a reference to the review rubric as needed.

##### Note: The code snippets from the notebook will be sometimes edited to preserve readibility. Please refer to the original for completeness.

#### 1. Compute camera calibration

#####  Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.
 Here, we explain how we transfrom this image:
 
 <img height=200 width=300 src="camera_cal/calibration1.jpg"
     alt="test"
     style="float: center;" />
     
into this one:
     
<img height=200 width=300 src="output_images/calibration1_undistorted.jpg"
     alt="test"
     style="float: center;" />
     

     
Or, taking into account the present context, this one:
     
<img height=200 width=300 src="pipeline_images/test6_original.jpg"
     alt="test"
     style="float: center;" />
     
into this one:
<img height=200 width=300 src="pipeline_images/test6_undistorted.jpg"
     alt="test"
     style="float: center;" />


In fact, the code dealing with this transformation is contained in the first two code cells of section 1. The relevant lines are:

    initialpoints =[] 
    targetpoints=[] 
    for file in list_fnames:
        file=os.path.join('camera_cal/',file)
        img=cv2.imread(file)
        * initialp=np.zeros((nx*ny,3),np.float32)
        * initialp[:,:2]=np.mgrid[0:9,0:6].T.reshape(-1,2)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ** ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None) 
    if ret==True :
        *** targetpoints.append(corners)
        *** initialpoints.append(initialp)
        c+=1
The lines with * are required in order to obtain the correct array to add the coordinates to the mappings (corners, initalp) given

The lines with ** deal with the extraction of corners to be mapped to the variable initial p.

The lines with *** deal with the addition of the mappings (corners,intialp) to the calibration array (targetpoints,initialpoints), in case there is a positive identification (as given by the line "if ret==True").

After the mappings are established, the calibration is obtained by the following lines:
        
        retval, cameraMatrix, distCoeffs, rvecs,cs =
        cv2.calibrateCamera(initialpoints, targetpoints,gray.shape[::-1],None,None)


#### 2. Apply a distortion correction to raw images.
##### Provide an example of a distortion-corrected image.
 Here, I used the undistort method of opencv with the parameters above calculated, and applied it to each image in the 'camera_cal' folder, saving it with the same name appended of the suffix "_undistorted" in the folder 'output_images'. The relevant lines are highlighted with *.
 
    path='output_images'
    for el in list_fnames:
        file=os.path.join('camera_cal/',el)
        img=cv2.imread(file)   
        * undist=cv2.undistort(img,cameraMatrix,distCoeffs,None,cameraMatrix)
        * file_sname= el[:-4]+ '_undistorted.jpg'
        plt.imsave(os.path.join(path,file_sname),undist)
        
 
#### 3. Use color transforms, gradients, etc., to create a thresholded binary image.
##### Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image. Provide an example of a binary image result.

We have picked the image 'pipeline_images/test6_undistorted.jpg' to design the pipeline.

In order to find a satisfying transformation we started by converting the image from RGB into the HLS colorspace. After some experimentation, we decided to use the S channel for gradient direction.

##### Note: The following 5 snippets are all contained in the function pipeline_grad_color_thresh defined in the auxilliary module 'aux_func.py'.
        
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:,:,2]
        
We then define a combined threshold based on directional gradient and color space:

First, we apply a scaledSobel transformation in the x direction, which allow us to detect vertical lines. This is given by.
        
    sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
Next, we defined the threshold gradient using the parameter 'sx_thresh', as follows:

    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

Next, we define the color thresholds as:

    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
Finally we stack the resulting channels together with a 'zero' channel and return the result:

    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    return color_binary

After hand-picking the parameters

     result = pipeline_grad_color_thresh(image_test_und,s_thresh=(50, 180), sx_thresh=(10, 190))
    
we obtain the image

<img height=200 width=300 src="pipeline_images/color_gradient.jpg"
     alt="test"
     style="float: center;" />

The next step is to select a candidate to the binary image. We have produced two. First, we use the line     

    gray=cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    
which outputs

<img height=200 width=300 src="pipeline_images/stacked_gray.jpg"
     alt="stacked_gray"
     style="float: center;" />
     
 We also considered a binary image where the activated pixels are correspond to those that fulfill both conditions. This is obtained with the help of the lines
 
    sxbinary=result[:,:,1]
    scbinary=result[:,:,2]
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(scbinary == 255) & (sxbinary == 255)] = 1,
    
which yield 
    <img height=200 width=300 src="pipeline_images/combined_gray.jpg"
     alt="combined_gray"
     style="float: center;" />

After inspection, we selected the first of the last two images, to apply a perspective transform.

   
    
    



#### 4. Apply a perspective transform to rectify binary image ("birds-eye view").
##### Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Here, we started by defining and drawing the polygon where the perspective transform is going to be applied. We used the lines

    x_l=700
    y_l=720
    src=np.float32([[300,680],[600,442],[720,442],[1200,680]])
    dst= np.float32([[150,y_l],[150,0],[150+x_l,0],[150+x_l,y_l]])
to define it and used the draw function of cv as follows:

    line_1=cv2.line(gray,tuple(src[0]),tuple(src[1]),(255,0,0),1)
    line_2=cv2.line(gray,tuple(src[1]),tuple(src[2]),(255,0,0),1)
    line_3=cv2.line(gray,tuple(src[2]),tuple(src[3]),(255,0,0),1)
    line_4=cv2.line(gray,tuple(src[3]),tuple(src[0]),(255,0,0),1)

The resulting image is as follows:

    <img height=200 width=300 src="pipeline_images/gray_to_scan.jpg"
     alt="combined_gray"
     style="float: center;" />

With the polygon rightly identified, we obtain the transformation matrix M

    M=cv2.getPerspectiveTransform(src,dst)
    
and apply the warpPerspective function to the gray object.

    warped=cv2.warpPerspective(gray,M,gray.shape[::-1],flags=cv2.INTER_LINEAR).

The output of this transformation is the following:

  <img height=200 width=300 src="pipeline_images/gray_perspective.jpg"
     alt="gray per"
     style="float: center;" />
     
and this shows the transformation corresponds to what is expected.


#### 5. Detect lane pixels and fit to find the lane boundary.


Here, before applying the detection algorithm, we did some pre-proccesing to the perspective image above. Namely, we have:



##### a) Eliminated shadows: This was accomplished by selecting a threshold, and setting all the pixels below that threshold to zero, by means of the following lines:

    gray[gray<threshold]=0

    
##### b) Set all the pixels around an image to zero: This is to clear out any details outside the scanning area that could interfere with the fitting of the polynomial.

    margin=30
    warped_filter[:,:150-margin]=0
    warped_filter[:,850+margin:]=0
    
The resulting perspective image from applying a) and b) is:

<img height=200 width=300 src="pipeline_images/gray_perspective_filtered.jpg"
     alt="gray per"
     style="float: center;" />
     


##### Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

For this task, I applied the function  find_lane_pixels(binary_warped) that can be found in the 'aux_functions' module, and that corresponds to my implementation of the quiz in the classes. Given the size of the function definition, we refer the reader to the module, where the function is properly documented, and whose image output is:

<img height=200 width=300 src="pipeline_images/lanes_id_pol.jpg"
     alt="gray per"
     style="float: center;" />



#### 6. Determine the curvature of the lane and vehicle position with respect to center.

##### Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

This task was accomplished with the help of the function measure_curvature_distance_real in the module 'aux_functions', which takes the inputs

binary_warped:The binary threshold image

leftx: The x points to fit a polynomial to the  the left lane

rightx: The x points to fit a polynomial to the  the right lane

righty: The y points to fit a polynomial to the right lane

lefty: The y points to fit a poynomial to the left lane


x_l: the width of the scanning region

y_l: the length of the scanning region

and outputs:

left_curverad: The estimated value for the radios of curvature of the left lane

right_curverad: The estimated value for the radios of curvature of the right lane

distance: The estimated distance between the center of the image and the center of the lane

binary_warped: The image without the lane fittings

Illustrating for the left-lane, its curvatures is caclulated with the help of the lines,

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    y_eval = np.max(ploty)*ym_per_pix
    left_fit= np.polyfit(lefty*ym_per_pix,leftx*xm_per_pix,2)
    right_fit= np.polyfit(righty*ym_per_pix,rightx*xm_per_pix,2)
    left_curverad = (1+(2*left_fit[0]*y_eval+left_fit[1])**2)**(3/2)/(2*np.abs(left_fit[0]))
    right_curverad = (1+(2*right_fit[0]*y_eval+right_fit[1])**2)**(3/2)/(2*np.abs(right_fit[0])) 

where,
     
     xm_per_pix = 3.7/x_l
     ym_per_pix = 30/y_l
     



Where lines 5-6 are just the implementation of the formula given in the classes. Note that we first re-fit the polynomial, this time with the rescaled coordinates.

As to the distance, we start by defining the coordinates

    center_x=binary_warped.shape[1]//2
    center_y=binary_warped.shape[0]-1
    
 Note that center_y is the height at which the camera is when we assume that it is at the base of the image.

while the center of the lane is the output of the following five lines,

    left_fit= np.polyfit(lefty,leftx,2)
    right_fit= np.polyfit(righty,rightx,2)
  
    left_midx = left_fit[0]*center_y**2 + left_fit[1]*center_y + left_fit[2]
    right_midx = right_fit[0]*center_y**2 + right_fit[1]*center_y + right_fit[2]
    mid_lane_x=(right_midx+left_midx)//2
    
where on lines 1-2 we refit the polynomial, on lines 3-4 we calculate the image of each of the polynomials for the point center_y, and the last line we take the average of those two images.

The distance is then 
        
        distance=np.abs(mid_lane_x - center_x) * xm_per_pix
where 
        
        xm_per_pix = 3.7/x_l
        
#### 7. Warp the detected lane boundaries back onto the original image.


##### Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.
 We obtain the following image with the line
 
     unwarped_back=cv2.warpPerspective(out_img,M,gray.shape[::-1],flags=cv2.WARP_INVERSE_MAP)
     
     
<img height=200 width=300 src="pipeline_images/warp_inverse.jpg"
     alt="gray per"
     style="float: center;" />
     
We can overlay this image with the original image with the line

    resulting=cv2.addWeighted(unwarped_back,0.5,image_test_und,0.8,0)

and see in the following image the fit of the lanes

<img height=200 width=300 src="pipeline_images/warped_inv_original_overlay.jpg"
     alt="gray per"
     style="float: center;" />
   
#### 8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position. 

We can now print the curvature and distance values. I used the lines

    cv2.putText(resulting, "L-lane curvature: "+ str(round(left_curverad,1)) + "m", (750, 50), cv2.FONT_ITALIC, 1.1,  (255, 255, 255), thickness=3, lineType=cv2.LINE_AA)
    cv2.putText(resulting, "R-lane curvature: "+ str(round(right_curverad,1)) + "m", (750, 100), cv2.FONT_ITALIC, 1.1,  (255, 255, 255), thickness=3, lineType=cv2.LINE_AA)
    cv2.putText(resulting, "Distance to mid-lane: "+ str(round(distance,2)) + "m", (750, 150), cv2.FONT_ITALIC, 1.1,  (255, 255, 255), thickness=3, lineType=cv2.LINE_AA)

which produce the image


<img height=200 width=300 src="pipeline_images/pipeline_output.jpg"
     alt="gray per"
     style="float: center;" />
     
Both estimates for the left and right ROC look ok.

####  9. Process the video.

Here, defined a pipeline function that can be seen in the first code cell of section 9 in the video. I then processed the project video with the help of moviepy, and obtained the following:
    
##### Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!)

<video height=960 width=540 controls src="project_video_output.mp4"
     alt="gray per"
     style="float: center;" />


We can see that the overall result was quite satisfactory.

##### Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?

# Please refer to the notebook. We can see that that algorithm breaks if we try to implement it in the challenge video. I have tried to come around that problem with a re-implementation based on the use of smaller windows around the previous frame. This implementation did not brake when the new pipeline was applied to the challenge video. However, the lane identification performed very poorly from the begining, given the presence of the bridge. One possible way of improving this implementation would be to take averages of frames and see the changes in the pixel values, and from there try to identify objects that do not make part of the lanes.

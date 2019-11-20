
# Advanced Lane Finding Project

Below steps are followed for this project:

1. Camera Calibration
2. Gradients & Color Transforms
3. Perspective Transform ("birds-eye view").
4. Detect lane pixels and fit to find the lane boundary.
5. Determine the curvature of the lane and vehicle position with respect to center.
6. Inverse Transform
7. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/1_undistorted.png "Undistorted"
[image2]: ./output_images/2_Thresholded.png "Thresholded"
[image3]: ./output_images/3_Perspective_Transform.png "Birds-Eye View"
[image4]: ./output_images/4_Histogram.png "Histogram"
[image5]: ./output_images/5_Lane_Prediction.png "Sliding Window"
[image6]: ./output_images/6_Polyfit.png "Polyfit"
[image7]: ./output_images/7_Inverse_Transform.png "Transform"
[image8]: ./output_images/8_Final_Result.png "Final Pineline"
[video1]: ./project_video_output.mp4 "Video"

#### 1.Camera Calibration

** Compute the camera calibration matrix and distortion coefficients given a set of chessboard images. Apply a distortion correction to raw images. **

There are 9 corners in a row and 6 corners in a column. I used glob to iterate over all the camera_cal images to extract the object and image points

```python
fnames = glob.glob("camera_cal/calibration*.jpg")

for fname in fnames:
    img = mpimg.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        
# use the object and image points to caliberate the camera and compute the camera matrix and distortion coefficients
ret, cameraMatrix, distortionCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_shape[:2],None,None)
```
Then I used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 
![alt text][image1]

#### 2. Gradients & Color Transforms
** Use color transforms, gradients, etc., to create a thresholded binary image. **
I used 2 kinds of gradient thresholds:

* Along the X axis.
* Directional gradient with thresholds of 30 and 90 degrees.

This is done since the lane lines are more or less vertical.

Then I applied following color thresholds:

* R & G channel thresholds so that yellow lanes are detected well.
* L channel threshold so that we don't take into account edges generated due to shadows.
* S channel threshold since it does a good job of separating out white & yellow lanes.
![alt text][image2]

#### 3. Perspective Transform
** Creating Birds-eye View **
After manually examining a sample image, I extracted the vertices to perform a perspective transform. The polygon with these vertices is drawn on the image for visualization. Destination points are chosen such that straight lanes appear more or less parallel in the transformed image.

```python
# Vertices extracted manually for performing a perspective transform
bottom_left = [220,720]
bottom_right = [1110, 720]
top_left = [570, 470]
top_right = [722, 470]

source = np.float32([bottom_left,bottom_right,top_right,top_left])

pts = np.array([bottom_left,bottom_right,top_right,top_left], np.int32)
pts = pts.reshape((-1,1,2))
copy = img.copy()
cv2.polylines(copy,[pts],True,(255,0,0), thickness=3)

# Destination points are chosen such that straight lanes appear more or less parallel in the transformed image.
bottom_left = [320,720]
bottom_right = [920, 720]
top_left = [320, 1]
top_right = [920, 1]

dst = np.float32([bottom_left,bottom_right,top_right,top_left])
M = cv2.getPerspectiveTransform(source, dst)
M_inv = cv2.getPerspectiveTransform(dst, source)
img_size = (image_shape[1], image_shape[0])

warped = cv2.warpPerspective(thresholded, M, img_size , flags=cv2.INTER_LINEAR)
    
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(copy)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(warped, cmap='gray')
ax2.set_title('Warped Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
```

![alt text][image3]

#### 4. Lane Detection
** Histogram **
The peaks in the histogram tell us about the likely position of the lanes in the image.

```python
histogram = np.sum(warped[np.int(warped.shape[0]/2):,:], axis=0)

# Peak in the first half indicates the likely position of the left lane
half_width = np.int(histogram.shape[0]/2)
leftx_base = np.argmax(histogram[:half_width])

# Peak in the second half indicates the likely position of the right lane
rightx_base = np.argmax(histogram[half_width:]) + half_width

print(leftx_base, rightx_base)
plt.plot(histogram)
```
![alt text][image4]
**Sliding Window Search **
I performed a sliding window search, starting with the likely positions of the 2 lanes, calculated from the histogram. I have used 10 windows of width 100 pixels.

The x & y coordinates of non zeros pixels are found, a polynomial is fit for these coordinates and the lane lines are drawn.

![alt text][image5]
** Searching around a previously detected line **
Since consecutive frames are likely to have lane lines in roughly similar positions, in this section we search around a margin of 50 pixels of the previously detected lane lines.
![alt text][image6]

#### 5. Calculate Radious of Curvature & Offset
The radius of curvature is computed according to the formula and method described in the classroom material. Since we perform the polynomial fit in pixels and whereas the curvature has to be calculated in real world meters, we have to use a pixel to meter transformation and recompute the fit again.

The mean of the lane pixels closest to the car gives us the center of the lane. The center of the image gives us the position of the car. The difference between the 2 is the offset from the center.
```python
def measure_radius_of_curvature(x_values):
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    # If no pixels were found return None
    y_points = np.linspace(0, num_rows-1, num_rows)
    y_eval = np.max(y_points)

    # Fit new polynomials to x,y in world space
    fit_cr = np.polyfit(y_points*ym_per_pix, x_values*xm_per_pix, 2)
    curverad = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
    return curverad

left_curve_rad = measure_radius_of_curvature(left_x_predictions)
right_curve_rad = measure_radius_of_curvature(right_x_predictions)
average_curve_rad = (left_curve_rad + right_curve_rad)/2
curvature_string = "Radius of curvature: %.2f m" % average_curve_rad
print(curvature_string)

# compute the offset from the center
lane_center = (right_x_predictions[719] + left_x_predictions[719])/2
xm_per_pix = 3.7/700 # meters per pixel in x dimension
center_offset_pixels = abs(img_size[0]/2 - lane_center)
center_offset_mtrs = xm_per_pix*center_offset_pixels
offset_string = "Center offset: %.2f m" % center_offset_mtrs
print(offset_string)
```
#### 6. Inverse Transform
** Warp the detected lane boundaries back onto the original image **
Highlited the area between two lanes using `cv2.fillPoly()` and performed inverse tranform.

![alt text][image7]
Result after constructing final pipeline:
![alt text][image8]
Output Video using `moviepy`
Here is the link for output [Project Video](./project_video_output.mp4) 

### Discussion
** Issues and Challenges **

####1. Gradient & Color Thresholding
* Lot of time is required to get optimum parameters for gradient and color channnel thresholding.
* The lanes lines in the challenge and harder challenge videos were extremely difficult to detect. They were either too bright or too dull. This prompted me to have R & G channel thresholding and L channel thresholding

####2. Bad Frames
* The challenge video has a section where the car goes underneath a tunnel and no lanes are detected
* To tackle this I had to resort to averaging over the previous well detected frames
* The lanes in the challenge video change in color, shape and direction. I had to experiment with color threholds to tackle this. Ultimately I had to make use of R, G channels and L channel thresholds.

Here is the link for output of [Challenge Video](./challenge_video_output.mp4)

### Areas of Improvement

The pipeline seems to fail for the harder challenge video. This video has sharper turns and at very short intervals.

* Take a better perspective transform: choose a smaller section to take the transform since this video has sharper turns and the lenght of a lane is shorter than the previous videos.
* Average over a smaller number of frames. Right now I am averaging over 12 frames. This fails for the harder challenge video since the shape and direction of lanes changes quite fast.

Here is the link for output of [Harder Challenge Video](./harder_challenge_video_output.mp4)

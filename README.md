## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image0]: ./output_images/camera_calib.png "Camera Calibration"
[image1]: ./output_images/chessboard_distor_cor_calibration1.jpg "Undistorted 1"
[image2]: ./output_images/distor_cor_test3.jpg "Undistorted"
[image3]: ./output_images/pers_trans_test3.jpg "Perspective Transform"
[image4]: ./output_images/threshold_test3.jpg "Thresholding"
[image5]: ./output_images/lane_finding_test3.jpg "Lane findings"
[image6]: ./output_images/curvature_test3.jpg "Curvature"
[image7]: ./etc/curvature_equation.png "Curvature Equation"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second code cell of the IPython notebook located in "./Project.ipynb". In this cell, I prepare object points(`objpoints`) that is the coordinates of the chessboard corners in the real world. Here I assume that the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image. `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. First, I convert a `BGR` image to a `GRAY` image using `cv2.cvtColor()` and `cv2.findChessboardCorners()`. Although I fail to process three images, it is enough to have as many points as I need. Here are examples.

![alt text][image0]

I then use the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. With this step, I obtain camera matrix (`mtx`) and distortion coefficients (`dist`). I can find an example image to be undistored in the following figure.

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one (`test3.jpg`). In `./helper.py`, `camera_calibration(img, objpoints, imgpoints, img_size)` provides an undistored image given `objpoints` and `imgpoints` as I mentioned above.

![alt text][image2]

#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspective_transform()`, which appears in lines 17 through 22 in the file `./helper.py`. The `perspective_transform()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
height, width = 720, 1280

src = np.float32([[225, 705], [550, 482], [732, 482], [1050, 705]])
margin = 400
dst = np.float32([[width/2 - margin, height],
                  [width/2 - margin, 0],
                  [width/2 + margin, 0],
                  [width/2 + margin, height]])
```

This resulted in the following source and destination points:

| Source         | Destination   |
|:--------------:|:-------------:|
| 225, 705       | 240,   720    |
| 550, 482       | 240,      0   |
| 732, 482       | 1040,     0   |
| 1050, 705      | 1040,   720   |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 25 through 50 in `./helper.py`):

```python
def thresholding(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    # HLS image
    img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l_channel = img_hls[:, :, 1]
    s_channel = img_hls[:, :, 2]

    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    # Absolute x derivative to accentuate lines away from horizontal
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) &
             (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    return combined_binary
```

First, I covert `BGR` iamge to `HLS` one and I apply sobel filter for `x` directional gradient to `L`-channel of the image. Its min and max thresold values are 20 and 100. Then I use `S`-channel for color
 transform with 170 and 255 threshold parameters. Finally I combine these two binary images (`sxbinary` and `s_binary`) into `combined_binary`. Here's an example of my output for this step.

![alt text][image4]


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

With given binary image by color and gradient thresholding, I use sliding window method which is introduced in the Udacity class with margin (100), minimum pixel (50) and 9 windows. This step appears in line 130 through 160 in `./helper.py`. It returns `leftx, lefty, rightx, righty` that are left and right lane pixel positions.

Next, I fit pixels to second order polynomial by `np.polyfit( , , 2)` for each lane. Here is an example.

![alt text][image5]

When I have previous lanes' information, I use the coefficients of polynomial for next lane findings instead of sliding window mehtod. `search_around_poly()` in `./helper.py` uses the previous coefficients with margin (100).

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 53 through 59 in my code in `./helper.py`. Here I assume that `x = A * y**2 + B * Y + C`. I evaluate curvature by the following definition with `y` (640). In addition, the position of the vehicle with respect to center is the middle of the bottom of lanes pixels.

```
python
center = undist.shape[1]/2 - (left_fit_x[0] + right_fit_x[0]) / 2
```

![alt text][image7]



#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 208 through 232 in my code in `./helper.py` in the function `fit_polynomial()`. Here is an example of my result on a test image (`test3.jpg`):

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In my opinion, my algorithm is sensitive to several factors First, the condition of road such as crack and different color match makes my algorithm to fail to find lanes. Second, it fails under the shadow of object above road such as a bridge and a telephone pole.

I think that there are two apporaches to have more robust algorithm. First, the goverment set a standard of road infrastructure that is very clean without cracks or break spots. Lastly, I can add another techs such as another gradient thresholding, Hough transform, a way to update recent n lanes, and advanced slding window method.

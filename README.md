**Vehicle Detection Project**

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_example.png
[image3]: ./output_images/test1_sliding_window.jpg
[image3-2]: ./output_images/test2_sliding_window.jpg
[image3-3]: ./output_images/test3_sliding_window.jpg
[image3-4]: ./output_images/test4_sliding_window.jpg
[image3-5]: ./output_images/test5_sliding_window.jpg
[image4]: ./output_images/test1_output_bboxes.jpg
[image4-2]: ./output_images/test2_output_bboxes.jpg
[image4-3]: ./output_images/test3_output_bboxes.jpg
[image4-4]: ./output_images/test4_output_bboxes.jpg
[image4-5]: ./output_images/test5_output_bboxes.jpg
[image5]: ./output_images/test1_heat_maps.jpg
[image5-2]: ./output_images/test2_heat_maps.jpg
[image5-3]: ./output_images/test3_heat_maps.jpg
[image5-4]: ./output_images/test4_heat_maps.jpg
[image5-5]: ./output_images/test5_heat_maps.jpg
[image6]: ./output_images/test1_heat_measurement.jpg
[image6-2]: ./output_images/test2_heat_measurement.jpg
[image6-3]: ./output_images/test3_heat_measurement.jpg
[image6-4]: ./output_images/test4_heat_measurement.jpg
[image6-5]: ./output_images/test5_heat_measurement.jpg
[image7]: ./output_images/test1_bounding_box.jpg
[image7-2]: ./output_images/test2_bounding_box.jpg
[image7-3]: ./output_images/test3_bounding_box.jpg
[image7-4]: ./output_images/test4_bounding_box.jpg
[image7-5]: ./output_images/test5_bounding_box.jpg
[video1]: https://youtu.be/7KxaXNs4rh0

---
###Writeup / README

The code for this step is contained in lines #66 through #81 of the file called `detection.py`.  

I started by reading in all the `vehicle` and `non-vehicle` images (lines# 41 through lines # 44 of vehicleclassifier.py).  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

I selected my parameters by optimizing my VehicleClassifier test accuracy. I experimented with different parameters for accuracy in my strawman/Example.ipynb.
My final parameters are:
```
color_space='YCrCb',  # RGB, HSV, LUV, HLS, YUV, YCrCb
orient=8,
pix_per_cell=8,
cell_per_block=2,
hog_channel='ALL',  # 0, 1, 2, or "ALL"
spatial_size=(16, 16),
hist_bins=32,
test_size=0.2,

```

Trained a linear SVM classifier in vehicleclassifier.py in the VehicleClassifier class.
The class takes a dictionary of parameters upon initialization which allowed me to iterate through different combinations of parameters and select the highest test accuracy. 
I didn't use a validation set which is not good practice and the hold out set effectively bled into my training set by doing this parameter selection using test set.
Since the model was generalizing well with my final parameters from Example.ipynb file, I could use it as is.

I used four set of sliding windows (details in config.py under process['window_parameters']) in a narrow vertical band (400-580 pixels) and with 75% overlap. 
After running some experiments these numbers gave me the least amount of false positives. 
However the large overlap generated many more likely boxes that I countered with a higher threshold. I chose the narrow vertical band to only detect the most likely area that cars may appear in (e.g. not in the sky). This choice will backfire if the car is traveling up or down steep hills.

![alt text][image3]

Searched using YCrCb 3-channel HOG features with spatially binned color and histograms of color in the feature vector. Here are some example images:

![alt text][image4]
![alt text][image4-3]
![alt text][image4-4]
![alt text][image4-5]
---
The Main.py is ran to generate the final video as shown below.
Here's a [link to my video result](https://youtu.be/7KxaXNs4rh0)


Function find_center_box() (line# 260 in detection.py) calculates center, width, height, and the score for each box that is identified (initialized at 1). The function average_boxes (line 332 in detection.py) uses these values and a history from previous frames to determine if a box should be retained given its score (which is accumulated over the history).
Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:
![alt text][image5]
![alt text][image5-2]
![alt text][image5-3]
![alt text][image5-4]
![alt text][image5-5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]
![alt text][image6-3]
![alt text][image6-4]
![alt text][image6-5]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]
![alt text][image7-3]
![alt text][image7-4]
![alt text][image7-5]

---

###Discussion

I left the detected boxes (red) in so we can see how the algorithm detects a lot of boxes and then only shows a subset (blue) based on the detection methodology described earlier. The false positive rate is pretty high which is not ideal. The training dataset is not very large and we could probably do a better job of model selection and parameter tuning. I mentioned earlier that I didn't do cross-validation and used the testing data in my parameter selection which is not a good practice. It stands to reason that the current model will not generalize to conditions outside this setting and lighting, different road conditions, and changing scenery would gravely impact the detection results. The filtering algorithm is helping to some extent but I thing the model will be easily confused (will probably detect pedestrian, animals, landmarks as vehicles).
The model doesn't separate two cars very well when they are close together.

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt2d
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter


# Define basic parameters
begin_frame = 1  # the first frame to start the detection
last_frame = 1050  # the last frame to end the detection, must be larger or equal to begin_frame
frame_interval = 1  # interval between each processed frame
begin_frame -= (begin_frame == last_frame)
total_extracted_frame = int(np.floor((last_frame - begin_frame) / frame_interval) + 1)  # calculate total frame
thickness_sample = 20  # sample size to calculate the mean thickness at each frame
pixel_interval = 3  # interval/distance between each sample in terms of pixel
detected_noise = 0  # detected noise is zero initially unless noise is found
noise_threshold = 2  # filter will be applied if the number of detected noise is larger/equal to the value we set

# Cropping, filter, and detecting options
cropping_image = 0  # set to 1 if we want to recrop the original image
pixel_millimeter_conversion = 0  # recalculate pixel and millimeter conversion if set to 1
detecting_middle_point = 1  # detect the middle point of the tube if set to 1
median_filter = 0  # apply median filter in the whole process despite noise detection result if set to 1
show_image = 0  # show images of each frame if set to 1
show_difference_detector = 0  # show detected edges from different methods if set to 1

# Define arrays
thickness_millimeter = np.zeros(total_extracted_frame)
sample_upper_part = np.zeros(thickness_sample)
frame_with_noise = []
noise_location = []

# Read the video data
v = cv2.VideoCapture("Second attempt.MP4")

# Get the first frame for cropping and millimeter per pixel calculation
ret, sample_frame = v.read()
image_variable = sample_frame

if cropping_image == 1:
    rect = cv2.selectROI(image_variable)
else:
    rect = (766, 907, 198, 299)

# Find millimeter per pixel
if pixel_millimeter_conversion == 1:
    J_measurement, rect_measurement = cv2.selectROI(image_variable)
    image_measurement = image_variable[int(rect_measurement[1]):int(rect_measurement[1] + rect_measurement[3]),
                                       int(rect_measurement[0]):int(rect_measurement[0] + rect_measurement[2])]
    cv2.imshow("Image Measurement", image_measurement)
    vertical_distance = abs(rect_measurement[3] - rect_measurement[1])
    millimeter_per_pixel = 70 / vertical_distance
else:
    millimeter_per_pixel = 0.117415730337079

# Start processing the video frame by frame
for i in range(total_extracted_frame):
    v.set(cv2.CAP_PROP_POS_FRAMES, begin_frame + (i - 1) * frame_interval)
    ret, frames = v.read()
    image_variable = frames

    # Crop the image
    image_cropped = image_variable[int(rect[1]):int(rect[1] + rect[3]), int(rect[0]):int(rect[0] + rect[2])]

    # Convert to grayscale and increase contrast
    image_gray = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2GRAY)
    image_stretched = cv2.equalizeHist(image_gray)

    # Detecting edge without a filter
    detected_edge = cv2.Canny(image_stretched, 100, 200)

    # Find the middle point
    range_val = 10
    row, col = detected_edge.shape

    if i == 0 and detecting_middle_point == 1:
        mid_column = col // 2
        left_edge = []
        right_edge = []

        # Find left edge
        for m in range(range_val):
            for n in range(row):
                if detected_edge[n, mid_column + m * 4] == 255:
                    left_edge.append(n)
                    break

        # Find right edge
        for m in range(range_val):
            for n in range(row - 1, -1, -1):
                if detected_edge[n - 10, mid_column + m * 4] == 255:
                    right_edge.append(n - 10)
                    break

        # Remove outliers
        mean_left_edge = np.mean(left_edge)
        mean_right_edge = np.mean(right_edge)
        center_row = round((mean_right_edge + mean_left_edge) / 2)
    elif i == 0 and detecting_middle_point == 0:
        center_row = 150

    # Measure thickness
    detected_noise = 0
    detected_noise_filter = 0
    detected_jump = 0
    detected_jump_filter = 0
    noise_location = []

    for k in range(thickness_sample):
        for j in range(row):
            if detected_edge[center_row + (k - thickness_sample // 2) * pixel_interval, 80 + j] == 255:
                sample_upper_part[k] = j + 80

                if i > 0 and sample_upper_part[k] + 8 < mean_upper_part:
                    detected_noise += 1
                    noise_location.append([center_row + (k - thickness_sample // 2) * pixel_interval, 80 + j])
                elif i > 0 and sample_upper_part[k] - 5 > mean_upper_part:
                    detected_jump += 1
                break

    # Apply median filter if noise is detected
    if median_filter == 1 or detected_noise >= noise_threshold:
        frame_with_noise.append(begin_frame + (i - 1) * frame_interval)
        detected_edge = cv2.medianBlur(image_gray, 5)
        detected_edge = cv2.Canny(detected_edge, 100, 200)

        # Measure thickness after median filter again
        for k in range(thickness_sample):
            for j in range(row):
                if detected_edge[center_row + (k - thickness_sample // 2) * pixel_interval, 80 + j] == 255:
                    sample_upper_part[k] = j + 80

                    if i > 0 and sample_upper_part[k] + 8 < mean_upper_part:
                        detected_noise_filter += 1
                    elif i > 0 and sample_upper_part[k] - 5 > mean_upper_part:
                        detected_jump_filter += 1
                    break

    # Computing upper and lower percentile before removing the outlier
    if median_filter == 1 or detected_noise >= noise_threshold:
        lower_percent = detected_noise_filter / thickness_sample * 100
        upper_percent = (1 - detected_jump_filter / thickness_sample) * 100
    else:
        lower_percent = detected_noise / thickness_sample * 100
        upper_percent = (1 - detected_jump / thickness_sample) * 100

    # Find average thickness after removing the outlier at this frame
    centralized_upper_part = np.delete(sample_upper_part, np.where(np.logical_or(sample_upper_part + 8 < mean_upper_part, sample_upper_part - 5 > mean_upper_part)))
    mean_upper_part = np.mean(centralized_upper_part)
    thickness_pixel = mean_upper_part - 48

    # Set the first frame as a reference frame
    if i == 0:
        Baseline = millimeter_per_pixel * thickness_pixel
        thickness_millimeter[i] = 0
    else:
        thickness_millimeter[i] = millimeter_per_pixel * thickness_pixel - Baseline

        if abs(thickness_millimeter[i] - thickness_millimeter[i - 1]) > 0.4:
            thickness_millimeter[i] = thickness_millimeter[i - 1]

    print(i)

    # Detect the edge with different detectors if needed
    if show_difference_detector == 1:
        detected_edge1 = cv2.Sobel(image_stretched, cv2.CV_64F, 1, 0, ksize=5)
        detected_edge2 = cv2.Sobel(image_stretched, cv2.CV_64F, 0, 1, ksize=5)
        detected_edge3 = cv2.Laplacian(image_stretched, cv2.CV_64F)
        detected_edge4 = cv2.Canny(image_stretched, 100, 200)
        detected_edge5 = cv2.Scharr(image_stretched, cv2.CV_64F, 1, 0)
        detected_edge6 = cv2.Scharr(image_stretched, cv2.CV_64F, 0, 1)

    # Show processing images
    if show_image == 1:
        plt.subplot(231)
        plt.imshow(cv2.cvtColor(image_cropped, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')

        plt.subplot(232)
        plt.imshow(image_gray, cmap='gray')
        plt.title('Grey Image')

        plt.subplot(234)
        plt.imshow(image_stretched, cmap='gray')
        plt.title('Stretched Grey Image')

        plt.subplot(235)
        plt.imshow(detected_edge, cmap='gray')
        plt.title('Detected Edge by Canny Method')

        if median_filter == 1 or detected_noise >= noise_threshold:
            plt.subplot(233)
            plt.imshow(cv2.cvtColor(detected_edge, cv2.COLOR_BGR2RGB))
            plt.title('Filtered Image')

        plt.show()

    # Show images from different methods
    if show_difference_detector == 1:
        plt.figure()
        plt.subplot(252)
        plt.imshow(np.abs(detected_edge1), cmap='gray')
        plt.title('Sobel Method')

        plt.subplot(253)
        plt.imshow(np.abs(detected_edge2), cmap='gray')
        plt.title('Prewitt Method')

        plt.subplot(254)
        plt.imshow(np.abs(detected_edge3), cmap='gray')
        plt.title('Laplacian Method')

        plt.subplot(257)
        plt.imshow(np.abs(detected_edge4), cmap='gray')
        plt.title('Canny Method')

        plt.subplot(258)
        plt.imshow(np.abs(detected_edge5), cmap='gray')
        plt.title('Scharr Method (Horizontal)')

        plt.subplot(259)
        plt.imshow(np.abs(detected_edge6), cmap='gray')
        plt.title('Scharr Method (Vertical)')

        plt.show()

# Plotting the graph: Thickness vs Frame
frame_axis = np.linspace(begin_frame, last_frame, total_extracted_frame)
plt.figure()
plt.plot(frame_axis, thickness_millimeter, linewidth=1)
plt.xlim(begin_frame, float('inf'))
plt.xlabel('Frame', fontsize=14)
plt.ylabel('Segment Thickness [mm]', fontsize=14)
plt.title('Thickness vs Frame', fontsize=14)
plt.grid(True)
plt.show()

# Plotting the graph: Thickness vs t
t = np.linspace((begin_frame - 1) / 29.97, (begin_frame - 1 + (total_extracted_frame - 1) * frame_interval) / 29.97, len(thickness_millimeter))
plt.figure()
plt.plot(t, thickness_millimeter, linewidth=1)
plt.xlim(t[0], float('inf'))
plt.xlabel('t [sec]', fontsize=14)
plt.ylabel('Segment Thickness [mm]', fontsize=14)
plt.title('Thickness vs t', fontsize=14)
plt.grid(True)
plt.show()

# Smooth data
B = np.convolve(thickness_millimeter, np.ones(5) / 5, mode='same')
plt.figure()
plt.plot(t, B, linewidth=2.8)
plt.xlim(t[0], float('inf'))
plt.xlabel('t [sec]', fontsize=14)
plt.ylabel('Segment Thickness [mm]', fontsize=14)
plt.title('Smoothed Thickness vs t', fontsize=14)
plt.grid(True)
plt.show()

# Exporting the figures (you can save them as needed)

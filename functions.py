import cv2
import numpy as np
import math
import time


def find_max(val1, val2):
    # Calculate maximum
    return val1 if val1 > val2 else val2

def find_min(val1, val2):
    # Calculate minimum
    return val1 if val1 < val2 else val2

def sum_fct(matrix):
    # Initialize the sum to 0
    result = 0

    # Iterate over each element in the matrix and accumulate the sum
    for row in matrix:
        for element in row:
            result += element

    return result


# custom quicksort implementation
def quicksort2(arr, axis=0):
    # if array is empty or has only one element
    if len(arr) <= 1:
        return arr
    
    # middle element pivot 
    pivot = arr[len(arr) // 2, axis]
    
    # partition the array into elements less than, equal to, and greater than the pivot
    left = [x for x in arr if x[axis] < pivot]
    middle = [x for x in arr if x[axis] == pivot]
    right = [x for x in arr if x[axis] > pivot]
    
    # recursively sort the left and right partitions and concatenate the results
    return quicksort2(left, axis) + middle + quicksort2(right, axis)

# 3D MEDIAN FILTER
def filtreMed(img, vois):
    # getting the shapes
    h, w, _ = img.shape
    imgMed = np.zeros(img.shape, img.dtype)

    # iterating through image pixels
    for y in range(h):
        for x in range(w):
            # keeping the image borders
            if y < int(vois/2) or y > int(h-vois/2) or x < int(vois/2) or x > int(w-vois/2):
                imgMed[y, x, :] = img[y, x, :]  # Keep the original pixel value
            else:
                # getting the pixel's neighbours matrix
                imgV = img[int(y-vois/2):int(y+vois/2), int(x-vois/2):int(x+vois/2), :]
                # creating t to store neighborhood pixels
                t = np.zeros((vois*vois, img.shape[2]), np.uint8)
                # storing them
                for yv in range(imgV.shape[0]):
                    for xv in range(imgV.shape[1]):
                        t[yv*vois+xv, :] = imgV[yv, xv, :]
                # sorting t
                t = np.array(quicksort2(t, axis=0))
                # getting the median
                imgMed[y, x, :] = t[int(vois*vois/2)+1, :]

    return imgMed



def gaussianFilter2(img, sigma):
    # getting the shapes
    h, w, c = img.shape

    imgGaussian = np.zeros(img.shape, img.dtype)
    kernel_size = 3 
    # Gaussian kernel
    gaussian_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    # normalizing
    gaussian_kernel = gaussian_kernel / 16
    # kernel center
    kernel_center = kernel_size // 2

    # iterating through image pixels
    for y in range(h):
        for x in range(w):
            for z in range(c):
                # calculating the pixel neighborhood boundaries
                y_start = max(y - kernel_center, 0)
                y_end = min(y + kernel_center + 1, h)
                x_start = max(x - kernel_center, 0)
                x_end = min(x + kernel_center + 1, w)
                # getting the pixel's neighbours matrix
                imgV = img[y_start:y_end, x_start:x_end, z]
                # calculating the weighted sum between the pixels matrix and kernel
                weighted_sum = np.sum(imgV * gaussian_kernel[:y_end - y_start, :x_end - x_start])
                imgGaussian[y, x, z] = round(weighted_sum)

    return imgGaussian



def min_max(r, g, b):
    h, w = r.shape

    max_val = np.zeros((h, w), dtype=np.uint8)
    min_val = np.zeros((h, w), dtype=np.uint8)
    
    # getting the max between red blue and green
    for i in range(h):
        for j in range(w):
            red, green, blue = r[i, j], g[i, j], b[i, j]

            max_val[i, j] = red if red > green else green if green > blue else blue
            min_val[i, j] = red if red < green else green if green < blue else blue

    return max_val, min_val

def rgb_to_hsv(rgb_frame):
    r, g, b = rgb_frame[:,:,0], rgb_frame[:,:,1], rgb_frame[:,:,2]

    max_val, min_val = min_max(r, g, b)

    # normalizing so it can be between 0 and 1 
    r, g, b = r / 255.0, g / 255.0, b / 255.0

    v = max_val / 255.0

    # Calculate saturation (S)
    s = ((v != 0) * (v - min_val / 255.0) / v) + (v == 0) * 0
    ('''If v is not equal to 0, the saturation (s) is calculated using the specified formula. This ensures that saturation is only calculated when the color is not black.

If v is equal to 0 (black color), the saturation (s) is set to 0''')

    # Calculate hue (H)
    delta = max_val - min_val
    
    # Create boolean masks
    mask_delta_not_zero = delta != 0 #This creates a boolean mask where each element is True if the corresponding element in delta is not equal to zero.
    mask_max_val_r = max_val == r * 255
    mask_max_val_g = max_val == g * 255
    mask_max_val_b = max_val == b * 255
    # These create boolean masks where each element is True if the corresponding element 
    # if max_val is equal to the corresponding element in r, g, or b multiplied by 255, respectively.

    # Calculate h without using np.select
    h = (mask_delta_not_zero * ((g - b) / delta % 6) * mask_max_val_r +  #% 6 is applied to ensure that the resulting hue is within the range [0, 6), aligning with the cyclical representation of hues around a color wheel. 
     mask_delta_not_zero * ((b - r) / delta + 2) * mask_max_val_g + #The addition of 2 corresponds to the fact that green is typically located at 120 degrees on a color wheel.
     mask_delta_not_zero * ((r - g) / delta + 4) * mask_max_val_b + #The + 4 is added to shift the hue range for the blue channel. The addition of 4 corresponds to the fact that blue is typically located at 240 degrees on a color wheel.
     ~mask_delta_not_zero * 0) * 60
  
    # If delta is not equal to 0, select the appropriate formula based on which color channel has the maximum value,
    # calculate the hue, and scale it by 60.
    # If delta is equal to 0 (color is a shade of gray), set the hue to 0.
    
    
    
    # Convert hue to the range [0, 360]
    h = (h + 360) % 360

    # Converting to uint8
    h = (h / 2).astype(np.uint8)
    s = (s * 255).astype(np.uint8)
    v = (v * 255).astype(np.uint8)

    hsv_frame = np.stack([h, s, v], axis=-1)
   
    return hsv_frame


def create_mask(hsv_frame, lower_threshold, upper_threshold):
    h, s, v = hsv_frame[:, :, 0], hsv_frame[:, :, 1], hsv_frame[:, :, 2]

    h_condition = (lower_threshold[0] <= h) & (h <= upper_threshold[0])
    s_condition = (lower_threshold[1] <= s) & (s <= upper_threshold[1])
    v_condition = (lower_threshold[2] <= v) & (v <= upper_threshold[2])

    mask = h_condition & s_condition & v_condition

    mask = mask.astype(np.uint8) * 255 #converting the boolean mask to a binary mask 

    return mask


def apply_mask_to_background(black_background, mask):
    

    # Using the mask to select pixels from the black background to detect the color that i want to detect
    result = black_background[:]
    result[mask == 0] = 0

    return result



def detect_color(frame, color_id, color_ranges):
    # Convert RGB to HSV
    hsv_frame = rgb_to_hsv(frame)

    # Get the color range for the specified color_id
    lower_threshold, upper_threshold = color_ranges.get(color_id, (0, 0))

    # Create a mask for the specified color range
    mask = create_mask(hsv_frame, lower_threshold, upper_threshold)

    # Apply the mask to a black background
    black_background = np.zeros_like(frame, dtype=np.uint8)
    result = apply_mask_to_background(black_background, mask)

    # Overlay the detected color as white on the black background
    result[mask > 0] = [255, 255, 255]

    return result


def detect_and_draw_color(frame, color_dict, detect_color):
    # Convert RGB to HSV
    hsv_frame = rgb_to_hsv(frame)

    # Create a black background
    black_background = np.zeros_like(frame, dtype=np.uint8)

    if detect_color in color_dict:
        color_range = color_dict[detect_color]
        # Create mask for the specified color range
        mask = create_mask(hsv_frame, *color_range[:2])

        # Apply the mask to the black background
        result = apply_mask_to_background_1(black_background, mask, color_range[2])

        # Overlay the detected color as white on the black background
        result[mask > 0] = color_range[2]

        # Overlay the detected color as circles on the original frame
        frame = apply_mask_to_background_1(frame, mask, color_range[2])

        return frame, result
    
    
def apply_mask_to_background_1(image, mask, color):
    # Find non-zero pixels in the mask
    non_zero_pixels = np.column_stack(np.where(mask > 0))

    # Draw a circle around each non-zero pixel
    # Set pixels in the image to the specified color
    for pixel in non_zero_pixels:
        image[pixel[0], pixel[1]] = color

    return image
    

# Function for background subtraction
def subtract_background(frame, background):
    
     # turning the image into hsv 
    hsv_frame = rgb_to_hsv(frame)
    # turning the background taken into hsv 
    hsv_background = rgb_to_hsv(background)

    diff = np.abs(hsv_frame - hsv_background) #the intensity diffrence between the frame and background 
    threshold_value = 50
    thresh = (diff[:, :, 2] > threshold_value).astype(np.uint8) * 255

    kernel_size = 5
    kernel_type = 0  # Change this to 0, 1, or 2 for RECT, CROSS, or ELLIPSE
    kernel = kernelInit(kernel_type, kernel_size)

    thresh = dilationFilter(thresh, kernel) #expanding the regions of white pixels in a binary image to create a more connected and filled object
    thresh = erosionFilter(thresh, kernel) #thin down the regions of white pixels

    return thresh

def detect_color_sub(frame, color_id, color_ranges,black_background):
    # Convert RGB to HSV
    hsv_frame = rgb_to_hsv(frame)

    # Get the color range for the specified color_id
    lower_threshold, upper_threshold = color_ranges.get(color_id, (0, 0))

    # Create a mask for the specified color range
    mask = create_mask(hsv_frame, lower_threshold, upper_threshold)

    # Apply the mask to a black background
    # black_background = np.zeros_like(frame, dtype=np.uint8)
    result = apply_mask_to_background(black_background, mask)

    # Overlay the detected color as white on the black background
    # result[mask > 0] = [255, 255, 255]

    return result



# FILTEEERS

# MEAN FILTER
def meanFilter(img, vois):
        # getting the shape
        h, w = img.shape
        imgMoy = np.zeros(img.shape, img.dtype)
        # iterating through rows
        for y in range(h):
            # iterating through cols
            for x in range(w):
                # keeping the image borders
                if y < int(vois/2) or y > int(h - vois/2) or x < int(vois/2) or y > int(w - vois/2):
                    imgMoy[y, x] = img[y, x]
                # getting pixel's neighbours matrix
                else:
                    imgV = img[int(y - vois/2):int(y + vois/2), int(x - vois/2):int(x + vois/2)]
                    # replacing by the mean
                    imgMoy[y, x] = np.mean(imgV)
        return imgMoy

# QUICK SORT
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return np.concatenate((quicksort(left), middle, quicksort(right)))

# MEDIAN FILTER
def medFilter(img, vois):
    # getting the shape
    h, w = img.shape
    imgMed = np.zeros(img.shape, img.dtype)
    # iterating through rows
    for y in range(h):
        # iterating through cols
        for x in range(w):
            # keeping the image borders
            if y < int(vois/2) or y > int(h - vois/2) or x < int(vois/2) or y > int(w - vois/2):
                imgMed[y, x] = img[y, x]
            # getting the pixel's neighbours matrix
            else:
                imgV = img[int(y - vois/2):int(y + vois/2), int(x - vois/2):int(x + vois/2)]
                t = np.zeros((vois*vois), np.uint8)
                for yV in range(imgV.shape[0]):
                    for xV in range(imgV.shape[1]):
                        t[yV*vois+xV] = imgV[yV, xV]
                sorted_t = quicksort(t) # sorting the values
                imgMed[y, x] = sorted_t[int(vois*vois/2)+1]
    return imgMed

# BINARIZATION FILTER
def binFilter(img, threshold, typeAlg):
    # getting the shape
    h, w = img.shape
    imgBin = img
    for y in range(h):
        # iterating through rows
        for x in range(w):
            # iterating through cols
            if typeAlg == 0:
                if img[y, x] < threshold:
                    imgBin[y, x] = 0
                else:
                    imgBin[y, x] = 255
            # threshold binary inverse
            elif typeAlg == 1:
                if img[y, x] < threshold:
                    imgBin[y, x] = 255
                else:
                    imgBin[y, x] = 0
            # threshold truncature
            elif typeAlg == 2:
                if img[y, x] > threshold:
                    imgBin[y, x] = threshold
            # threshold to zero
            elif typeAlg == 3:
                if img[y, x] < threshold:
                    imgBin[y, x] = 0
            # threshold to zero inverse
            elif typeAlg == 4:
                if img[y, x] > threshold:
                    imgBin[y, x] = 0

    return imgBin

# GAUSSIAN FILTER

def gaussianKernel(size, sigma):
    # kernel size must be odd
    if size % 2 == 0:
        return None
    # creating the kernel
    kernel = np.zeros((size, size), dtype=float)
    # kernel center
    center = size // 2

    for i in range(size):
        for j in range(size):
            # distance between the element and the center
            x, y = i - center, j - center
            # gaussian formula
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    # normalising the kernel
    kernel /= np.sum(kernel)
    return kernel

def gaussianFilter(img, gkernel):
    # getting the shape
    h, w = img.shape
    imgGaussian = np.zeros(img.shape, img.dtype)
    # kernel center
    kernel_center = gkernel.shape[0] // 2

    for y in range(h):
        for x in range(w):
            # keeping the image borders
            if y < kernel_center or y > h - kernel_center or x < kernel_center or x > w - kernel_center:
                imgGaussian[y, x] = img[y, x]
            # getting the pixel's neighbours matrix
            else:
                imgV = img[y - kernel_center:y + kernel_center + 1, x - kernel_center:x + kernel_center + 1]
                weighted_sum = 0
                # calculating the weighted sum between the pixels matrix and kernel
                for i in range(imgV.shape[0]):
                    for j in range(imgV.shape[1]):
                        weighted_sum += imgV[i, j] * gkernel[i, j]
                imgGaussian[y, x] = round(weighted_sum)

    return imgGaussian

# LAPLACIAN FILTER

def laplacianFilter(img):
    # getting the shape
    h, w = img.shape
    imgLaplacian = np.zeros_like(img, dtype=np.uint8)
    # laplacian kernel
    kernel = np.array([[1, 4, 1],
                       [4, -20, 4],
                       [1, 4, 1]])
    kernel_center = 1
    
    # iterating through image pixels
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            pixel_sum = 0
            # iterating through kernel
            for i in range(3):
                for j in range(3):
                # calculating the weighted sum between the pixels matrix and kernel
                    pixel_sum += img[y + i - 1, x + j - 1] * kernel[i, j]
            # clipping between 0 and 255
            if pixel_sum < 0:
                      imgLaplacian[y, x] = 0
            elif pixel_sum > 255:
                      imgLaplacian[y, x] = 255
            else:
                      imgLaplacian[y, x] = pixel_sum
                    
    #imgLaplacian = imgLaplacian[kernel_center:kernel_center+h, kernel_center:kernel_center+w]

    return imgLaplacian


def kernelInit(k, size):
    # kernel size must be odd
    if size % 2 == 0:
        return None
    # RECT
    if k == 0:
        kernel = np.ones((size, size))
    # CROSS
    elif k == 1:
        kernel = np.zeros((size, size))
        kernel[size//2,:] = 1 # middle row
        kernel[:,size//2] = 1 # middle column
    # ELLIPSE
    else:
        kernel = np.zeros((size, size))
        kernel[1:size-1, :] = 1  # all lines receive 1 besides the first and the last one
        kernel[:, size//2] = 1  # the middle column receive 1

    return kernel

# EROSION FILTER
def erosionFilter(img, kernel):
    # getting the shape
    h, w = img.shape
    # kernel center
    kernel_center = kernel.shape[0] // 2
    result = np.zeros_like(img)
    # iterating through image pixels
    for y in range(h):
        for x in range(w):
            min_val = 255  # assuming 8-bit grayscale image
            for i in range(kernel.shape[0]):
                for j in range(kernel.shape[0]):
                        # checking if the neighbors are in the image
                        if 0 <= y + i - kernel_center < h and 0 <= x + j - kernel_center < w:
                            # checking if the kernel element is active
                            if kernel[i, j] == 1:
                                # update min comparing with the corresponding neighbour
                                min_val = min(min_val, img[y + i - kernel_center, x + j - kernel_center])
            result[y, x] = min_val

    return result

# DILATION FILTER
def dilationFilter(img, kernel):
    # getting the shape
    h, w = img.shape
    # kernel center
    kernel_center = kernel.shape[0] // 2
    result = np.zeros_like(img)
    # iterating through image pixels
    for y in range(h):
        for x in range(w):
            max_val = 0  # assuming 8-bit grayscale image
            for i in range(kernel.shape[0]):
                for j in range(kernel.shape[0]):
                        # checking if the neighbors are in the image
                        if 0 <= y + i - kernel_center < h and 0 <= x + j - kernel_center < w:
                            # checking if the kernel element is active
                            if kernel[i, j] == 1:
                                # update max comparing with the corresponding neighbour
                                max_val = max(max_val, img[y + i - kernel_center, x + j - kernel_center])
            result[y, x] = max_val

    return result

# OPENING FILTER
def openingFilter(img, kernel):
    return erosionFilter(dilationFilter(img, kernel), kernel)

# CLOSING FILTER
def closingFilter(img, kernel):
    return dilationFilter(erosionFilter(img, kernel), kernel)

# EMBOSS FILTER
def embossFilter(img):
    # getting the shape
    h, w = img.shape
    img_emboss = np.zeros(img.shape, dtype=float)
    # emboss kernel
    kernel = np.array([[-2, -1, 0],
                       [-1,  1, 1],
                       [ 0,  1, 2]])

    # calculating the weighted sum between the pixels matrix and kernel
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            img_emboss[y, x] = np.sum(img[y-1:y+2, x-1:x+2] * kernel)

    min, max = 255, 0
    # getting min & max
    for y in range(h):
        for x in range(w):
            if img_emboss[y, x] > max:
                max = img_emboss[y, x]
            if img_emboss[y, x] < min:
                min = img_emboss[y, x]
    # normalizing between 0 and 255
    img_emboss = (img_emboss - min) / (max - min) * 255

    return img_emboss.astype(np.uint8)


# MOTION FILTER
def motionFilter(img, kernel_size):

    # getting the shape
    h, w = img.shape
    img_blurred = np.zeros_like(img, dtype=np.float32)
    # motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size) # middle row
    kernel /= kernel_size # normalizing
    kernel_center = kernel_size // 2
    # iterating through the image pixels
    for y in range(h):
        for x in range(w):
            # keeping the image borders
            if y < kernel_center or y > h - kernel_center or x < kernel_center or x > w - kernel_center:
                img_blurred[y, x] = img[y, x]
            # getting the pixel's neighbours matrix
            else:
                imgV = img[y - kernel_center:y + kernel_center + 1, x - kernel_center:x + kernel_center + 1]
                weighted_sum = 0
                # calculating the weighted sum between the pixels matrix and kernel
                for i in range(imgV.shape[0]):
                    for j in range(imgV.shape[1]):
                        weighted_sum += imgV[i, j] * kernel[i, j]
                img_blurred[y, x] = weighted_sum

    return img_blurred.astype(np.uint8)

# SYMMETRIC Transformation
def symmetricFilter(img):
    h, w = img.shape
    imgS = np.copy(img)
    # iterating through pixels
    for y in range(h):
        for x in range(w):
            # horizontal transformation (w - x - 1 is the mirror pixel of x)
            imgS[y, x] = img[y, w - 1 - x]

    return imgS

def invisibility_cloak():
    cap = cv2.VideoCapture(0)
    # attendre 3s pour que la caméra soit prete
    time.sleep(3)
    background = 0

    # sauvegarder une image apres les 3s pour servir de background
    ret, background = cap.read()
    
    # inverse horizontalement l'image du fond
    background = np.flip(background, axis=1)

    while cap.isOpened():
        ret, frame = cap.read()
        # inverse horizontalement le frame
        frame = np.flip(frame, axis=1)

        # convertit le frame en HSV
        hsv_frame = rgb_to_hsv(frame)

        # définit les bornes inférieure et supérieure pour nos couleurs
        lower_blue = np.array([0, 120, 100]) 
        upper_blue = np.array([40, 255, 255])

        lower_green = np.array([40, 50, 80])
        upper_green = np.array([80, 100, 255])

        # crée un masque pour identifier la couleur dans l'image
        mask = create_mask(hsv_frame, lower_blue, upper_blue)

        # remplace la couleur detectée par le background
        frame[np.where(mask == 255)] = background[np.where(mask == 255)]
        cv2.imshow('Display', frame)

        # Check for the 'q' key press to exit the loop
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    
    
def green_screen(path_to_image):
    
    cap = cv2.VideoCapture(0)
    
    # Read a frame from the camera
    ret, frame = cap.read()

    # read the replacement image
    replacement_img = cv2.imread(path_to_image)
    
    # get dimensions of our frame to resize the background image
    frame_height, frame_width, _ = frame.shape
    
    # get dimensions of replacement image
    replacement_height, replacement_width, _ = replacement_img.shape
    
    # scaling factors between frame and replacement image
    scale_x = frame_width / replacement_width
    scale_y = frame_height / replacement_height
    
    replacement_resized = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    
    # resize the replacement image to match the frame size
    for y in range(frame_height):
        for x in range(frame_width):
            repl_x = int(x / scale_x)
            repl_y = int(y / scale_y)
            # copy pixels from replacement image to resized image
            if repl_x < replacement_width and repl_y < replacement_height:
                replacement_resized[y, x] = replacement_img[repl_y, repl_x]
    
    while cap.isOpened():
        ret, img = cap.read()
        # flip the frame horizontally
        img = np.flip(img, axis=1)
        
        # convert the frame to HSV color space
        hsv_frame = rgb_to_hsv(img)
        
        # define the lower and upper bounds for our colors
        lower_green = np.array([40, 50, 80])
        upper_green = np.array([80, 100, 255])
        lower_blue = np.array([0, 120, 100]) 
        upper_blue = np.array([40, 255, 255])
        
        # create a mask to identify the color in the frame
        mask = create_mask(hsv_frame, lower_blue, upper_blue)
        
        # replace the color in the frame with the resized replacement image
        img[np.where(mask == 255)] = replacement_resized[np.where(mask == 255)]
        
        cv2.imshow('Display', img)
        
        # Check for the 'q' key press to exit the loop
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

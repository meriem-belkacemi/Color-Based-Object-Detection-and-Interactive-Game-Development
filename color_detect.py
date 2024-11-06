import cv2
import numpy as np


def detect_color(frame, lower_threshold, upper_threshold):
    """
    Detect color within a specified HSV range in a given frame.

    Parameters:
    - frame: Input image frame in BGR format.
    - lower_threshold: Lower HSV threshold values as a numpy array.
    - upper_threshold: Upper HSV threshold values as a numpy array.

    Returns:
    - success: True if color is detected, False otherwise.
    - color_position: Horizontal position of the detected color.
    """
    # Convert BGR to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask
    mask = create_mask(hsv_frame, lower_threshold, upper_threshold)

    # Find contours manually without using cv2
    contours = find_contours_from_scratch(mask)

    if contours:
        # Get the largest contour (assumed to be the detected color)
        largest_contour = max(contours, key=lambda x: len(x))

        # Get the bounding box of the contour
        x, y, w, h = bounding_rect(largest_contour)

        # Calculate the horizontal position of the detected color
        color_position = x + w // 2

        # Return the color position
        return True, color_position

    return False, None

def find_contours_from_scratch(mask):
    """
    Find contours in a binary mask without using cv2.findContours.

    Parameters:
    - mask: Binary mask indicating regions of interest.

    Returns:
    - contours: List of contours, where each contour is represented as a numpy array.
    """
    contours = []

    # Find white pixel indices in the binary image
    white_pixel_indices = zip(*np.where(mask == 255))

    # If the image is entirely black, there are no contours to find
    if not white_pixel_indices:
        return contours

    # Convert indices to contour points
    contour_points = [np.array([index[::-1]]) for index in white_pixel_indices]

    # Sort contour points by the first coordinate
    contour_points.sort(key=lambda x: x[0, 0])

    # Check if there are any contour points
    if not contour_points:
        return contours

    # Convert sorted points to a list of contours
    current_contour = [contour_points[0]]
    for point in contour_points[1:]:
        # Calculate distance between the current point and the last point in the current contour
        distance = np.linalg.norm(point - current_contour[-1][0])

        # If distance is greater than 1 (not a neighbor), start a new contour
        if distance > 1.0:
            contours.append(np.array(current_contour))
            current_contour = [point]
        else:
            current_contour.append(point)

    # Add the last contour to the list
    contours.append(np.array(current_contour))

    return contours

def bounding_rect(contour):
    """
    Compute the bounding rectangle for a given contour.

    Parameters:
    - contour: A numpy array representing a contour.

    Returns:
    - min_x: Minimum x-coordinate of the bounding rectangle.
    - min_y: Minimum y-coordinate of the bounding rectangle.
    - width: Width of the bounding rectangle.
    - height: Height of the bounding rectangle.
    """
    if contour is None or len(contour) == 0:
        return 0, 0, 0, 0

    # Ensure the contour is flattened if it's multi-dimensional
    if isinstance(contour[0], np.ndarray):
        contour = contour.squeeze()

    # Calculate the bounding rectangle coordinates
    if len(contour.shape) == 1:
        min_x = max_x = contour[0]
        min_y = max_y = contour[1]
    else:
        min_x = np.min(contour[:, 0])
        min_y = np.min(contour[:, 1])
        max_x = np.max(contour[:, 0])
        max_y = np.max(contour[:, 1])

    return min_x, min_y, max_x - min_x, max_y - min_y

def create_mask(hsv_frame, lower_threshold, upper_threshold):
    """
    Create a binary mask based on HSV color thresholds.

    Parameters:
    - hsv_frame: HSV representation of an image.
    - lower_threshold: Lower HSV threshold values as a numpy array.
    - upper_threshold: Upper HSV threshold values as a numpy array.

    Returns:
    - mask: Binary mask indicating the regions within the specified color range.
    """
    # Extract individual channels from the HSV frame
    h, s, v = hsv_frame[:, :, 0], hsv_frame[:, :, 1], hsv_frame[:, :, 2]

    # Apply conditions to each channel based on the provided thresholds
    h_condition = (lower_threshold[0] <= h) & (h <= upper_threshold[0])
    s_condition = (lower_threshold[1] <= s) & (s <= upper_threshold[1])
    v_condition = (lower_threshold[2] <= v) & (v <= upper_threshold[2])

    # Combine conditions to create the binary mask
    mask = h_condition & s_condition & v_condition

    # Convert the boolean mask to binary (0 or 255)
    mask = mask.astype(np.uint8) * 255

    return mask

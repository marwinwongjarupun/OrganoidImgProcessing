#import all modules and libraries needed for the project
import cv2 # Core OpenCV for image processing
import numpy as np # Array operations and mathematical functions
from matplotlib import pyplot as plt #  Plotting (not used in main workflow) (to check when coding)
from scipy import ndimage # Scientific image processing
from skimage import measure, color, io # Measures cell properties, to change channels (colour) and input and output images (io)

# Ask to input the file path
image_path = input("Please enter the path to your image file (e.g., 'path/to/droplet.jpg'): ")

# Read the image
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print("Error: Could not load image. Check the file path.")
else:
    print("Image loaded successfully!")

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to the grayscale image
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Apply Otsu's thresholding
_, image_thresh = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Detect contours
contours, hierarchy = cv2.findContours(image_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Display contours
image_display = image.copy()
cv2.drawContours(image_display, contours, -1, (255, 0, 0), 1)
plt.imshow(cv2.cvtColor(image_display, cv2.COLOR_BGR2RGB))
plt.title("Filtered Contours")
plt.axis('off')
plt.show()


def organoidsize(contour_list):
    num_contours = len(contour_list)
    print("Number of contours detected:", num_contours)

    if num_contours == 4:
        # Only one organoid detected
        area = cv2.contourArea(contour_list[-1])
        print("Area is: " + str(area) + " pixels squared")

        # Convert pixel area to diameter (approximate)
        diameter_pixels = 2 * (area / 3.1416) ** 0.5  # from area = πr²

        while True:
            microns_per_pixel_input = input("Enter your microscope's microns-per-pixel ratio (e.g., 0.5): ")
            if microns_per_pixel_input.strip():  # Check if input is not empty
                try:
                    microns_per_pixel = float(microns_per_pixel_input)
                    break  # Exit loop if conversion succeeds
                except ValueError:
                    print("Invalid input. Please enter a valid number (e.g., 0.5).")
            else:
                print("Input cannot be empty. Please enter a value.")

        diameter_microns = diameter_pixels * microns_per_pixel

        print("Diameter is: " + str(diameter_pixels) + " pixels")
        print("Diameter is: " + str(diameter_microns) + " microns")

        if 100 <= diameter_microns <= 150:
            print("Droplet sent to CHANNEL 1 (single organoid, correct size)")
        else:
            print("Droplet sent to CHANNEL 2 (single organoid, wrong size)")

    elif num_contours > 4:
        # Multiple organoids
        print("Droplet sent to CHANNEL 2 (multiple organoids)")

    else:
        # No organoids found
        print("Droplet sent to CHANNEL 2 (no organoids detected)")

    print("Contours detected:", len(contour_list))


# Using the function to analyze the contours:
organoidsize(contours)
                           
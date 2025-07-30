from ultralytics import YOLO
import os
import glob
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Load the trained model
model = YOLO("/Users/karthik/Desktop/ABIC - Sorter Project/runs/detect/organoid_detector16/weights/best.pt")
print("Model loaded successfully!")

# Path to validation images
val_images_path = "/Users/karthik/Desktop/trainingimgs/images/val/*.png"  # Using .png

# Get list of validation images
val_images = glob.glob(val_images_path)
if not val_images:
    print("Error: No validation images found in", val_images_path)
    exit(1)

# Run inference on validation images
results = model.predict(source=val_images, save=False, conf=0.4, save_txt=True, save_conf=True)
print("Predictions saved to runs/predict/labels/ (text files only)")

# Plot and display each image with bounding boxes
for i, result in enumerate(results):
    # Load the original image
    img_path = result.path
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for Matplotlib

    # Get image dimensions
    h, w = img.shape[:2]

    # Initialize counters
    organoid_count = 0
    droplet_count = 0

    # Count objects
    for box in result.boxes:
        class_id = int(box.cls)  # Class ID (0 for droplet, 1 for organoid)
        if class_id == 0:
            droplet_count += 1
        else:
            organoid_count += 1

    # Print counts and channel decision
    print(f"\nImage: {os.path.basename(img_path)}")
    print(f" - Organoids: {organoid_count}, Droplets: {droplet_count}")
    
    # Create a copy of the image for plotting
    plot_img = img.copy()

    if organoid_count == 1:
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Apply Gaussian blur to the grayscale image
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        # Apply Otsu's thresholding
        _, image_thresh = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Detect contours
        contours, hierarchy = cv2.findContours(image_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours on the copied image
        cv2.drawContours(plot_img, contours, -1, (0, 255, 0), 2)  # Green 

        
        # Calculate diameter (assuming circular shape)
        organoid_countour = contours[-1] # Assuming the last contour is the organoid
        perimeter = cv2.arcLength(organoid_countour, True)
        diameter = perimeter / np.pi  # Diameter from circumference
        
        pixel_micron_conversion  = float(input("Enter pixel to micron conversion factor: "))
        diameter_microns = diameter * pixel_micron_conversion 

        if diameter_microns > 100 and diameter_microns < 150:
            print("Send to channel 1")
        else:
            print("Send to channel 2")
    else:
        print("Send to channel 2")

    # Plot the image
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    ax = plt.gca()

    # Draw bounding boxes
    for box in result.boxes:
        class_id = int(box.cls)  # Class ID (0 for droplet, 1 for organoid)
        class_name = "droplet" if class_id == 0 else "organoid"  # Explicit mapping
        confidence = float(box.conf)  # Confidence score
        coords = box.xywh[0].tolist()  # [x_center, y_center, width, height]

        # Convert xywh to xyxy for plotting
        x_center, y_center, width, height = coords
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2

        # Draw rectangle
        rect = plt.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r' if class_name == "organoid" else 'b', facecolor='none')
        ax.add_patch(rect)

        # Add label with confidence
        label = f"{class_name} {confidence:.2f}"
        plt.text(x1, y1 - 10, label, color='r' if class_name == "organoid" else 'b', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

        # Print detection details
        print(f" - Detected: {class_name}, Confidence: {confidence:.3f}, Box: [{x_center:.2f}, {y_center:.2f}, {width:.2f}, {height:.2f}]")

    plt.title(f"Image: {os.path.basename(img_path)}")
    plt.axis('off')
    plt.show()  # Display the image
    plt.close()

print("All images displayed with bounding boxes.")


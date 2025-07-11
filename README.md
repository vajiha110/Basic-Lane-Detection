# Basic-Lane-Detection
# Code
import cv2
import numpy as np
import os

# Ensure the output folder exists
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Get all .jpg files in the current folder
image_files = [file for file in os.listdir('.') if file.endswith('.jpg')]

# Loop through each image
for image_file in image_files:
    print(f"🔄 Processing {image_file}...")

    image = cv2.imread(image_file)

    if image is None:
        print(f"❌ Error: Could not load {image_file}")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (0, height),
        (width, height),
        (width, int(height * 0.6)),
        (0, int(height * 0.6))
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(masked_edges, 2, np.pi / 180, 100, minLineLength=40, maxLineGap=5)
    line_image = np.zeros_like(image)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)

    combo = cv2.addWeighted(image, 0.8, line_image, 1, 1)

    # Save to output folder
    output_path = os.path.join(output_dir, f"output_{image_file}")
    cv2.imwrite(output_path, combo)
    print(f"✅ Saved: {output_path}")

    # Show the image
    cv2.imshow(f"Lane Detection - {image_file}", combo)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Create a function to save histogram as an image
def save_histogram(hist, title, filename):
    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(256), hist, color='blue', width=1)  # Use bar for histogram
    plt.title(title)
    plt.xlim([0, 256])
    plt.xlabel('Pixel value')
    plt.ylabel('Frequency')
    plt.grid()
    plt.savefig(filename)
    plt.close()


# Define a function to create the Gaussian and Laplacian pyramid images in a structured layout
def process(image):
    gaussian_pyramid = []
    laplacian_pyramid = []

    current_image = image.copy()
    original_height, original_width = image.shape[:2]

    for _ in range(3):
        # Apply Gaussian blur and downsample
        blurred = cv2.GaussianBlur(current_image, (5, 5), 0)
        downsampled = cv2.pyrDown(blurred)
        gaussian_pyramid.append(downsampled)

        # Upsample to the current image size and create the residual (Laplacian) image
        upsampled = cv2.pyrUp(
            downsampled, dstsize=(current_image.shape[1], current_image.shape[0])
        )
        residual = current_image.astype(int) - upsampled.astype(int)
        residual = cv2.add(residual, 128)
        laplacian_pyramid.append(residual)

        # Prepare the next iteration with the downsampled image
        current_image = downsampled

    # Resize all images to the same width for display
    display_images = []
    for i, (g, l) in enumerate(zip(gaussian_pyramid, laplacian_pyramid)):
        g_resized = cv2.resize(
            g, (original_width // (2 ** (i + 1)), original_height // (2 ** (i + 1)))
        )
        l_resized = cv2.resize(
            l, (original_width // (2 ** (i)), original_height // (2 ** (i)))
        )
        display_images.append((g_resized, l_resized))

    # Prepare a blank canvas for layout (size based on expected arrangement)
    canvas_height = original_height * 2
    canvas_width = original_width + sum(img.shape[1] for img, _ in display_images)
    canvas = np.ones((canvas_height, canvas_width), dtype=np.uint8) * 255

    # Place the original image at the top-left corner of the canvas
    canvas[0:original_height, 0:original_width] = image

    # Place the pyramid images in a grid pattern
    y_offset = original_width
    for i, (g_resized, l_resized) in enumerate(display_images):
        # Place Gaussian pyramid image
        g_height, g_width = g_resized.shape[:2]
        canvas[
            original_height - g_height : original_height, y_offset : y_offset + g_width
        ] = g_resized

        # Place Laplacian pyramid image next to it
        l_height, l_width = l_resized.shape[:2]
        canvas[
            original_height : original_height + l_height, y_offset - l_width : y_offset
        ] = l_resized

        # Add the third Gaussian approximation under the third Laplacian image
        if i == 2:  # Check if this is the third level
            canvas[
                original_height: original_height+g_height,
                y_offset : y_offset + g_width,
            ] = g_resized

        # Update the vertical offset for the next level
        y_offset += g_width
    
    #计算并保存原图和第一张残差图的直方图
    hist1, bin1 = np.histogram(image.flatten(), bins=256, range=[0, 256])
    hist2, bin2 = np.histogram(display_images[0][1].flatten(), bins=256, range=[0, 256])
    # Save the histograms
    save_histogram(hist1, 'Original Image Histogram', 'result\\1-2.jpg')
    save_histogram(hist2, 'First Residual Image Histogram', 'result\\1-3.jpg')
    return canvas


# Set the working directory and load the image
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
image_path = r"test images\\demo-1.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Process the image to get the final result
result = process(image)

# Save the result
cv2.imwrite("result\\1-1.jpg", result)



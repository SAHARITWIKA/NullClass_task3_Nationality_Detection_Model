import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image

def load_and_resize_image(image_path, target_size=(64, 64)):
    """
    Loads an image from path and resizes it to the given size.
    Returns the resized image as a NumPy array.
    """
    image = Image.open(image_path)
    image = image.resize(target_size)
    return np.array(image)

def extract_dominant_color(image, k=4):
    """
    Extracts the dominant color from an image using KMeans clustering.
    Returns the dominant color in RGB format as a tuple (R, G, B).
    """
    # Resize for faster processing
    img = cv2.resize(image, (64, 64))
    img = img.reshape((-1, 3))  # Flatten the image to a 2D array of pixels

    # KMeans to find dominant clusters
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(img)
    colors = kmeans.cluster_centers_
    labels = kmeans.labels_

    # Get the most frequent color cluster
    dominant_color = colors[np.bincount(labels).argmax()]
    return tuple(map(int, dominant_color))  # Convert float RGB to int

def predict_color_name(rgb):
    """
    Converts an RGB tuple into a readable color name.
    """
    r, g, b = rgb

    if r > 200 and g < 100 and b < 100:
        return "Red"
    elif r < 100 and g > 200 and b < 100:
        return "Green"
    elif r < 100 and g < 100 and b > 200:
        return "Blue"
    elif r > 200 and g > 200 and b < 100:
        return "Yellow"
    elif r > 200 and g > 200 and b > 200:
        return "White"
    elif r < 50 and g < 50 and b < 50:
        return "Black"
    elif 100 < r < 200 and 50 < g < 150 and 0 < b < 80:
        return "Brown"
    elif r > 160 and g > 130 and b > 130:
        return "Pink"
    elif r < 100 and g > 100 and b > 100:
        return "Cyan"
    elif r > 180 and g > 180 and b > 180:
        return "Grey"
    else:
        return "Other"

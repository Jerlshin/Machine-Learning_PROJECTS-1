import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image

def compress_image(image_path, K=16, max_iters=10):
    # Read the image using Pillow (PIL)
    original_img = Image.open(image_path)

    # Convert the image to a NumPy array
    X_img = np.array(original_img)

    # Reshape the image data to a 2D array
    height, width, channels = X_img.shape
    X_reshaped = X_img.reshape(height * width, channels)

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=K, max_iter=max_iters)
    kmeans.fit(X_reshaped)
    idx = kmeans.predict(X_reshaped)
    centroids = kmeans.cluster_centers_

    # Replace pixel values with the centroids' values to compress the image
    X_recovered = centroids[idx]
    X_recovered = X_recovered.reshape(height, width, channels)

    # Display the original and compressed images using Matplotlib
    fig, ax = plt.subplots(1, 2, figsize=(16, 16))
    ax[0].imshow(original_img)
    ax[0].set_title('Original')
    ax[0].axis('off')

    ax[1].imshow(X_recovered.astype(np.uint8))
    ax[1].set_title('Compressed with %d colors' % K)
    ax[1].axis('off')

    plt.show()

if __name__ == '__main__':
    image_path = r"E:\Courses\MachineLearning_DeepLearning.AI\C3 - Unsupervised Learning, Recommenders, Reinforcement Learning\week1\C3W1A\C3W1A1\bird_small.png"
    K = 16  # Number of clusters (colors) for compression
    max_iters = 10  # Maximum number of iterations for K-Means

    compress_image(image_path, K, max_iters)

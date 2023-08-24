import numpy as np
import matplotlib.pyplot as plt

def find_closest_centroids(X, centroids):
    k = centroids.shape[0]
    idx = np.zeros(X.shape[0], dtype=int)
    for i in range(X.shape[0]):
        distance = []
        for j in range(k):
            norm_ij = np.linalg.norm(X[i] - centroids[j])
            distance.append(norm_ij)
        idx[i] = np.argmin(distance)
    return idx
            

def compute_centroids(X, idx, K):
    m, n = X.shape
    centroids = np.zeros((k, n))
    
    for k in range(K):
        points = X[idx==k]
        centroids[k] = np.mean(points, axis=0)
    return centroids

def run_KMeans(X, initial_centroids, max_iters=10):
    m, n = X.shape
    K = initial_centroids.shape[0] # no of centroids
    
    centroids = initial_centroids
    previous_centroids = centroids
    
    idx = np.zeros(m)
    plt.figure(figsize=(8, 6))
    
    for i in range(max_iters):
        print("K-Means iteration %d/%d" % (i, max_iters-1))
        idx = find_closest_centroids(X, centroids)
    plt.show()
    return centroids, idx

def KMeans_init_centroids(X, K):
    randidx = np.random.permutation(X.shape[0])
    centroids = X[randidx[:K]]
    
    return centroids

## Image compression with K-Means

if __name__ == '__main__':
    original_img = plt.imread(r"E:\Courses\MachineLearning_DeepLearning.AI\C3 - Unsupervised Learning, Recommenders, Reinforcement Learning\week1\C3W1A\C3W1A1\bird_small.png")
    X_img = np.reshape(original_img, (original_img.shape[0] * original_img.shape[1], 3))
    K = 16 # 16 clusters
    max_iters = 10

    initial_centroids = KMeans_init_centroids(X_img, K)
    centroids, idx = run_KMeans(X_img, initial_centroids, max_iters)
    idx = find_closest_centroids(X_img, centroids)
    X_recovered = centroids[idx, :]
    X_recovered = np.reshape(X_recovered, original_img.shape)

    # Display original image
    fig, ax = plt.subplots(1,2, figsize=(16,16))
    plt.axis('off')

    ax[0].imshow(original_img)
    ax[0].set_title('Original')
    ax[0].set_axis_off()


    # Display compressed image
    ax[1].imshow(X_recovered)
    ax[1].set_title('Compressed with %d colours'%K)
    ax[1].set_axis_off()
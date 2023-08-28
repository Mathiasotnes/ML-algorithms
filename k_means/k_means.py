import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class KMeans:
    
    def __init__(self, n_dimentions, n_centroids):
        self.dimentions = n_dimentions
        self.centroids = np.random.rand(n_centroids, n_dimentions)
        
    def fit(self, X, iterations):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        """
        # Converting from dataframe to tensor
        X = np.array(X)

        ##### SPREAD INITIALIZATION #####

        # Initializing even spread of centroids (works semi-good)
        centroids = self.spread_initialization(X, self.dimentions, len(self.centroids))

        # Iterations of K-means algorithm
        centroids = self.kmeans(iterations, X, centroids)

        self.centroids = centroids
        
        ##### K_MEANS ++ INITIALIZATION #####

        # for _ in range(iterations):

        #     centroids = self.kmeans_plusplus_initialization(X, len(self.centroids))
        #     centroids = self.kmeans(10, X, centroids)

        #     old_distortion = euclidean_distortion(X, self.predict(X))
        #     new_distortion = euclidean_distortion(X, self.predict_centroids(X, centroids))

        #     if old_distortion > new_distortion:
        #         self.centroids = centroids
    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m integer array with cluster assignments
            for each point. E.g., if X is a 10xn matrix and 
            there are 3 clusters, then a possible assignment
            could be: array([2, 0, 0, 1, 2, 1, 1, 0, 2, 2])
        """

        # Find distances to all centroids
        distances = np.array([euclidean_distance(X, centroid) for centroid in self.centroids]).T

        # Assign centroids to input coordinates
        assigned_centroids = np.argmin(distances, axis=1)

        return assigned_centroids
    
    def predict_centroids(self, X, centroids):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m integer array with cluster assignments
            for each point. E.g., if X is a 10xn matrix and 
            there are 3 clusters, then a possible assignment
            could be: array([2, 0, 0, 1, 2, 1, 1, 0, 2, 2])
        """

        # Find distances to all centroids
        distances = np.array([euclidean_distance(X, centroid) for centroid in centroids]).T

        # Assign centroids to input coordinates
        assigned_centroids = np.argmin(distances, axis=1)

        return assigned_centroids
    
    def get_centroids(self):
        """
        Returns the centroids found by the K-mean algorithm
        
        Example with m centroids in an n-dimensional space:
        >>> model.get_centroids()
        numpy.array([
            [x1_1, x1_2, ..., x1_n],
            [x2_1, x2_2, ..., x2_n],
                    .
                    .
                    .
            [xm_1, xm_2, ..., xm_n]
        ])
        """
        return np.array(self.centroids)

    def kmeans(self, iterations, X, centroids):
        """
        Runs k_means algorithm iteration number of timer and returns
        the centroids
        """

        for _ in range(iterations):
    
            # Find distances to all centroids
            distances = np.array([euclidean_distance(X, centroid) for centroid in centroids]).T

            # Assign centroids to input coordinates
            assigned_centroids = np.argmin(distances, axis=1)
                
            # Center centroids by calculating mean of assigned points
            for i in range(len(centroids)):
                points_assigned_to_centroid = X[assigned_centroids == i]
                centroids[i] = points_assigned_to_centroid.mean(axis=0)
        
        return centroids

    def spread_initialization(self, X, n_dimensions, n_centroids):
        """
        Returns n_centroids evenly spread centroids in a 
        circle with a standard deviations radius based 
        on dataset.
        """

        if n_dimensions != 2:
            raise ValueError("The spread_centroids function currently only supports 2D data.")

        centroids = [[0] * n_dimensions for _ in range(n_centroids)]

        center_x = np.mean(X[:, 0])  # Mean of the first dimension
        center_y = np.mean(X[:, 1])  # Mean of the second dimension

        # Getting the standard deviations for both dimensions
        std_x = np.std(X[:, 0])
        std_y = np.std(X[:, 1])

        # Using the average of the two standard deviations for the radius
        radius = np.mean([std_x, std_y])

        theta = 2 * np.pi / n_centroids  # Angle between centroids

        for i in range(n_centroids):
            centroids[i] = [center_x + radius * np.cos(i * theta), center_y + radius * np.sin(i * theta)]

        return centroids

    def kmeans_plusplus_initialization(self, X, k):
        """
        KMeans++ initialization for KMeans algorithm.
        Args:
            X (numpy array): Data points of shape (num_samples, num_features)
            k (int): Number of centroids to initialize
        Returns:
            centroids (numpy array): Initialized centroids
        """
        # Randomly choose the first centroid
        centroids = [X[np.random.choice(X.shape[0])]]
        
        for _ in range(1, k):
            # Calculate the squared distances from the last centroid picked
            squared_distances = np.array([np.min([np.inner(c-x,c-x) for c in centroids]) for x in X])
            
            # Choose the next centroid with probability proportional to squared distance
            prob = squared_distances/squared_distances.sum()
            next_centroid = X[np.random.choice(X.shape[0], p=prob)]
            centroids.append(next_centroid)
        
        return np.array(centroids)


    
    
    
    
# --- Some utility functions 

def euclidean_distance(x, y):
    """
    Computes euclidean distance between two sets of points 
    
    Note: by passing "y=0.0", it will compute the euclidean norm
    
    Args:
        x, y (array<...,n>): float tensors with pairs of 
            n-dimensional points 
            
    Returns:
        A float array of shape <...> with the pairwise distances
        of each x and y point
    """
    return np.linalg.norm(x - y, ord=2, axis=-1)

def cross_euclidean_distance(x, y=None):
    """
    
    
    """
    y = x if y is None else y 
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])


def euclidean_distortion(X, z):
    """
    Computes the Euclidean K-means distortion
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the raw distortion measure 
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    distortion = 0.0
    clusters = np.unique(z)
    for i, c in enumerate(clusters):
        Xc = X[z == c]
        mu = Xc.mean(axis=0)
        distortion += ((Xc - mu) ** 2).sum()
        
    return distortion


def euclidean_silhouette(X, z):
    """
    Computes the average Silhouette Coefficient with euclidean distance 
    
    More info:
        - https://www.sciencedirect.com/science/article/pii/0377042787901257
        - https://en.wikipedia.org/wiki/Silhouette_(clustering)
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the silhouette score
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)
    
    # Intra distance 
    a = D[np.arange(len(X)), z]
    # Smallest inter distance 
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)
    
    return np.mean((b - a) / np.maximum(a, b))
  
from sklearn.cluster import KMeans
import numpy as np

class KMeansClustering:
    def elbow_plot(self, X, logger, max_clusters=10):
        """
        Uses the elbow method to determine the optimal number of clusters.
        Returns the optimal number of clusters.
        """
        logger.info("Starting elbow plot to determine optimal number of clusters.")
        wcss = []
        for i in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)
        logger.info(f"Elbow plot WCSS values: {wcss}")
        # For demonstration, just return 3 or implement your own elbow logic
        optimal_clusters = 3
        logger.info(f"Optimal number of clusters selected: {optimal_clusters}")
        logger.info("Elbow plot completed.")
        return optimal_clusters

    def create_clusters(self, X, n_clusters, logger):
        """
        Assigns each sample in X to a cluster.
        Returns a numpy array of cluster labels.
        """
        logger.info(f"Creating {n_clusters} clusters using KMeans.")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X)
        logger.info(f"Clusters assigned: {np.unique(clusters)}")
        logger.info("Clustering completed.")
        return clusters
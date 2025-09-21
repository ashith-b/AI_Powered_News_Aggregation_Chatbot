import numpy as np
import umap
import hdbscan
from loguru import logger
from config import Config

class NewsClustering:
    """Cluster news articles to detect duplicates and similar stories"""
    
    def __init__(self, 
                 n_components=Config.UMAP_N_COMPONENTS,
                 random_state=Config.UMAP_RANDOM_STATE, 
                 min_cluster_size=Config.HDBSCAN_MIN_CLUSTER_SIZE):
        self.n_components = n_components
        self.random_state = random_state
        self.min_cluster_size = min_cluster_size
        self.umap_model = None
        self.clusterer = None
    
    def fit_transform(self, embeddings):
        """Fit UMAP and HDBSCAN models and transform embeddings
        
        Args:
            embeddings: List of article embeddings
            
        Returns:
            np.array: Cluster assignments for each article
        """
        logger.info(f"Running dimensionality reduction with UMAP (n_components={self.n_components})")
        self.umap_model = umap.UMAP(
            n_components=self.n_components,
            random_state=self.random_state
        )
        reduced = self.umap_model.fit_transform(embeddings)
        
        logger.info(f"Clustering with HDBSCAN (min_cluster_size={self.min_cluster_size})")
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            prediction_data=True
        )
        clusters = self.clusterer.fit_predict(reduced)
        
        logger.info(f"Found {len(np.unique(clusters))} clusters")
        return clusters
    
    def add_clusters_to_df(self, df, embeddings):
        """Add cluster assignments to DataFrame
        
        Args:
            df: DataFrame with articles
            embeddings: List of article embeddings
            
        Returns:
            DataFrame: Original DataFrame with added 'cluster' and 'cluster_size' columns
        """
        clusters = self.fit_transform(embeddings)
        
        df = df.copy()
        df['cluster'] = clusters
        
        # Add cluster size for each article
        df['cluster_size'] = df.groupby('cluster')['id'].transform('count')
        
        return df 
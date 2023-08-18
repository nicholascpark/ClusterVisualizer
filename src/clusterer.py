import argparse
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import yaml
from io import BytesIO
from azure.storage.blob import ContainerClient, BlobServiceClient
from urllib.parse import urlparse
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid
import numpy as np

class Clustering:

    def __init__(self, clustering_config, data_config):

        self.config = clustering_config
        self.data_config = data_config
        self.cluster_features = self.data_config['columns']['features']['numeric'] + self.data_config['columns']['features']['categorical']
        self.model = None

    def cluster_data(self, data, clustering_method = 'kmeans'):

        print("Clustering using {}...".format(clustering_method))

        df = data[self.cluster_features ]

        if self.model is None:
            if clustering_method == 'kmeans':
                kmeans_config = self.config['kmeans']
                n_clusters = kmeans_config['n_clusters']
                self.model = KMeans(n_clusters=n_clusters, **kmeans_config['kwargs'])
                labels = self.model.fit_predict(df)
            elif clustering_method == 'hierarchical':
                hierarchical_config = self.config['hierarchical']
                n_clusters = hierarchical_config['n_clusters']
                self.model = AgglomerativeClustering(n_clusters=n_clusters, **hierarchical_config['kwargs'])
                labels = self.model.fit_predict(df)
            elif clustering_method == 'dbscan':
                dbscan_config = self.config['dbscan']
                eps = dbscan_config['eps']
                min_samples = dbscan_config['min_samples']
                self.model = DBSCAN(eps=eps, min_samples=min_samples, **dbscan_config['kwargs'])
                labels = self.model.fit_predict(df)
        else:
            labels = self.model.predict(df)
  
        data['cluster'] = labels
        print('Done!', labels)

        cluster_centers = self.model.cluster_centers_ if clustering_method in ['kmeans', 'hierarchical'] else None

        return data, cluster_centers
    
    def elbow_plot(self, data, max_cluster = 12, clustering_method = 'kmeans'):

        distortions = []
        K = range(1, max_cluster + 1)
        df = data[self.cluster_features ]
        for k in K:
            kmeanModel = KMeans(n_clusters=k)
            kmeanModel.fit(df)
            distortions.append(kmeanModel.inertia_)

        plt.figure(figsize=(16,8))
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method showing the optimal k')
        plt.show()

    def tune_hyperparameters(self, model, data):

        data = data[self.cluster_features ]

        if model == 'kmeans':
            param_grid = {'n_clusters': np.arange(2, 6)}#, 'init': ['k-means++', 'random'], 'algorithm': ['lloyd', 'elkan']}
        elif model == 'hierarchical':
            param_grid = {'n_clusters': np.arange(5, 11), 'linkage': ['ward', 'complete', 'average', 'single']}
        elif model == 'dbscan':
            param_grid = {'eps': np.linspace(0.1, 1, 10), 'min_samples': np.arange(1, 11)}
        else:
            raise ValueError('Invalid model')

        best_params = None
        best_score = -np.inf
        
        for params in ParameterGrid(param_grid):
            if model == 'kmeans':
                estimator = KMeans(**params)
            elif model == 'hierarchical':
                estimator = AgglomerativeClustering(**params)
            elif model == 'dbscan':
                estimator = DBSCAN(**params)
            else:
                raise ValueError('Invalid model')
                
            estimator.fit(data)
            labels = estimator.labels_
            score = silhouette_score(data, labels, n_jobs=-1)
            
            if score > best_score:
                best_score = score
                best_params = params
                self.model = estimator
            print(score, params)
                
        return best_params, best_score
        
    def describe_clusters(self, df, features = None):
       
        if features is None:
            features = [col.lower() for col in self.cluster_features]
        else:
            features = [col.lower() for col in features]

        df = df[features + ['cluster']]
       
        # Define a list to store data
        data = []

        # Loop through each of your clusters
        for cluster in df['cluster'].unique():
            # Subset the original DataFrame to create a new DataFrame with only the current cluster's data
            cluster_data = df[df['cluster'] == cluster]
            
            # Drop the 'cluster' column from this DataFrame
            cluster_data = cluster_data.drop('cluster', axis=1)
            
            # Calculate the basic statistics for this cluster and add to the list
            data.append(
                [
                    cluster, 
                    pd.Series([len(cluster_data)]*len(cluster_data.columns), index = cluster_data.columns),
                    # cluster_data.min(), 
                    cluster_data.quantile(0.25),
                    cluster_data.median(), 
                    cluster_data.quantile(0.75),
                    # cluster_data.max()
                ]
            )

        # Create a new dataframe with the aggregated data
        # cluster_summary = pd.concat([pd.DataFrame(data[i][1:], index=['count', 'min', 'q25', 'median', 'q75', 'max'], columns=df.columns[:-1]).assign(cluster=data[i][0]) for i in range(len(data))])
        cluster_summary = pd.concat([pd.DataFrame(data[i][1:], index=['count', 'q25', 'median', 'q75'], columns=df.columns[:-1]).assign(cluster=data[i][0]) for i in range(len(data))])

        # Pivot the dataframe to get the desired format
        cluster_summary = cluster_summary.set_index('cluster', append=True).stack().unstack('cluster').swaplevel().sort_index()
        return cluster_summary.round(3)


if __name__ == '__main__':
    
    mm = MailerScoringModel("../config/mailerscoring_config.yaml", data_source="DS_ONLY")

    # default_data_urL = "https://sae1devdsml.blob.core.windows.net/ds-labs/data/dp-dmg/processed_data/campaign_universe/CAMPAIGN_CODE=522/"
    default_data_url = "https://sae1devdsml.blob.core.windows.net/ds-labs/data/ml-mailer-model/v4/training/DMG_ONLY/"
    default_config_file = "../config/cluster.yml"

    parser = argparse.ArgumentParser(description='Perform clustering and visualization.')
    parser.add_argument('-m', '--method', choices=['kmeans', 'hierarchical', 'dbscan'], help='Clustering method (kmeans, hierarchical, dbscan)')
    parser.add_argument('-d', '--data_url', help='URL to the blob in storage', default = default_data_url)
    parser.add_argument('-c', '--config_file', help='Path to the configuration file (YAML)', default = default_config_file)
    parser.add_argument('-p', '--plot', choices=['2d', '3d'], help='Plot the clusters (3D plot if available)', default = '3d')
    parser.add_argument('-f', '--fraction', help='Fraction of data to use for clustering', default = 1.0)
    parser.add_argument('-u', '--unscaled', choices=['y','n'], help='Plot with unscaled data: y/n', default = 'n')
    args = parser.parse_args()

    config = yaml.load(open(args.config_file), Loader=yaml.FullLoader)

    cluster_and_visualize(
        clustering_method = args.method, 
        data_url = args.data_url, 
        config = config, 
        plot = args.plot, 
        fraction = args.fraction,
        plot_with_unscaled_features = args.unscaled == 'y'
    )
    plt.show()
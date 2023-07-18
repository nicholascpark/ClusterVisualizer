from argparse import ArgumentParser
from clusterer import Clustering
from visualizer import Visualizer
import pandas as pd
import argparse
import yaml
import matplotlib.pyplot as plt
from data_loader import DataLoader

def main(
        config,
        method,
        plot,
        fraction,
        plot_unscaled
    ):

    # Load data
    dataloader = DataLoader(config['data'])
    data = dataloader.extract_data(source_type = 'pickle', fraction = fraction)
    
    # Initialize clusterer
    c = Clustering(clustering_config = config['clustering'],    data_config = config['data'])
    v = Visualizer(visual_config     = config['visualization'], data_config = config['data'])

    c.elbow_plot(data)    

    print(data.head())

    # Perform clustering
    data, centers = c.cluster_data(data, method)
    
    # Visualize clusters
    v.visualize_clusters(data, plot, ['ACTUAL_DEBT_VALUE', 'ACTUAL_INCOME', 'AGE'], method, plot_unscaled, centers = centers, dimensionality_reduction = 'TSNE')

    v.plot_density_features(data, features = ['ACTUAL_DEBT_VALUE', 'ACTUAL_INCOME', 'AGE'])

    # Describe clusters
    cluster_stats = c.describe_clusters(data, columns = ['ACTUAL_DEBT_VALUE', 'ACTUAL_INCOME', 'AGE'])

    print("Cluster Statistics:\n", cluster_stats)



if __name__ == '__main__':
    
    default_data_url = "https://sae1devdsml.blob.core.windows.net/ds-labs/data/ml-mailer-model/v4/training/DMG_ONLY/"
    default_config_file = "./config/dmg_mailer_data.yml"

    parser = argparse.ArgumentParser(description='Perform clustering and visualization.')
    parser.add_argument('-c', '--config_file', help='Path to the configuration file (YAML)', default = default_config_file)
    parser.add_argument('-m', '--method', choices=['kmeans', 'hierarchical', 'dbscan'], help='Clustering method (kmeans, hierarchical, dbscan)', default = 'kmeans')
    parser.add_argument('-d', '--dimension', choices=['2d', '3d'], help='Plot the clusters (3D plot if available)', default = '3d')
    parser.add_argument('-f', '--fraction', help='Fraction of data to use for clustering', default = 1.0)
    parser.add_argument('-u', '--plot_unscaled', choices=['y','n'], help='Plot with unscaled data: y/n', default = 'n')
    args = parser.parse_args()

    config = yaml.load(open(args.config_file), Loader=yaml.FullLoader)

    main(
        config = config, 
        method = args.method,
        plot = args.dimension, 
        fraction = args.fraction,
        plot_unscaled = args.plot_unscaled == 'y'
    )
    plt.show()
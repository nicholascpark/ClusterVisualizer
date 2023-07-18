import argparse
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import yaml
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')
from io import BytesIO
from azure.storage.blob import ContainerClient, BlobServiceClient
from urllib.parse import urlparse
import seaborn as sns
from sklearn.manifold import TSNE

class Visualizer:

    def __init__(self, visual_config, data_config):

        self.config = visual_config
        self.data_config = data_config
        self.cluster_features = self.data_config['columns']['features']['numeric'] + self.data_config['columns']['features']['categorical']

    def __generate_TSNE(self, df, features, plot = '2d'):

        if features is None:
            features = [col.lower() for col in self.cluster_features]
        else:
            features = [col.lower() for col in features]

        # Use t-SNE for dimensionality reduction
        n_components = 2 if plot == '2d' else 3
        tsne = TSNE(n_components = n_components, random_state = 0)
        X_tsne = tsne.fit_transform(df[features])
        df[['TSNE_{x}' for x in range(1, n_components + 1)]] = X_tsne

        return  df[['TSNE_{x}' for x in range(1, n_components + 1)]]

    def boxplot_features(self, df, features):

        if features is None:
            features = [col.lower() for col in self.cluster_features]
        else:
            features = [col.lower() for col in features]

        df = df[features + ['cluster']]

        # Define the number of rows and columns for your subplot grid
        n = len(df.columns) - 1  # number of features
        ncols = len(df['cluster'].unique())  # number of clusters
        nrows = n

        # Create a figure and axes with a size based on your number of plots
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 5*nrows))

        for idx, feature in enumerate(features):  # Exclude 'cluster'
            for c in df['cluster'].unique():
                # Select the axis where the plot will be drawn
                ax = axs[idx, c]
                # Draw the plot for the current feature and cluster
                sns.boxplot(y=df[df['cluster'] == c][feature], ax=ax)
                # Set the title for the plot
                ax.set_title(f'Cluster {c}, Feature: {feature}')

        plt.tight_layout()
        plt.show()

    def plot_density_features(self, df, features):

        if features is None:
            features = [col.lower() for col in self.cluster_features]
        else:
            features = [col.lower() for col in features]

        df = df[features + ['cluster']]

        nrows = len(df.columns) - 1  # number of features

        # Create a figure and axes with a size based on your number of plots
        fig, axs = plt.subplots(nrows=nrows, figsize=(10, 8*nrows))

        for idx, feature in enumerate(df.columns[:-1]):  # Exclude 'cluster'
            ax = axs[idx]
            
            for c in df['cluster'].unique():
                # Draw the density plot for the current feature and cluster
                sns.kdeplot(df[df['cluster'] == c][feature], ax=ax, fill=True, alpha=0.5, label=f'Cluster {c}')
            
            # Set the title and legend for the plot
            ax.set_title(f'Feature: {feature}')
            ax.legend()

        plt.tight_layout()
        plt.show()

    def visualize_clusters(self, data, plot, features, clustering_method, plot_unscaled, centers, dimensionality_reduction = None):

        if dimensionality_reduction is not None:
            assert dimensionality_reduction in ['TSNE', 'PCA'], "Dimensionality Reduction Method must be either TSNE or PCA"

        print("Plotting Unscaled Version:", plot_unscaled)

        x_label, y_label, z_label = self.config['x_label'], self.config['y_label'], self.config['z_label']

        if plot_unscaled:
            x_label, y_label, z_label = x_label.lower(), y_label.lower(), z_label.lower()
        if dimensionality_reduction == 'TSNE':
            tsne_data = self.__generate_TSNE(data, features, plot)
            x_label, y_label, z_label = tsne_data.columns[0], tsne_data.columns[1], tsne_data.columns[2]

    
        labels = data['cluster']
        
        fig = plt.figure(figsize = (12,8))

        if plot == '3d':

            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(data[x_label], data[y_label], data[z_label], c=labels, cmap='viridis')
            ax.set_zlabel(z_label)
            ax.scatter(centers[:,0], centers[:,1], centers[:,2], marker='*', c='black', s=150)

            xmin, xmax = data.iloc[:, 0].min(), data.iloc[:, 0].max()
            ymin, ymax = data.iloc[:, 1].min(), data.iloc[:, 1].max()
            zmin, zmax = data.iloc[:, 2].min(), data.iloc[:, 2].max()
            
            xpad, ypad, zpad = (xmax - xmin) * 0.1, (ymax - ymin) * 0.1, (zmax - zmin) * 0.1
            ax.set_xlim(xmin - xpad, xmax + xpad) 
            ax.set_ylim(ymin - ypad, ymax + ypad) 
            ax.set_zlim(zmin - zpad, zmax + zpad)

        else:

            plt.scatter(data[x_label], data[y_label], c=labels, cmap='viridis')

        t = fig.suptitle(self.config['title'] + " with "+ clustering_method, fontsize=14)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()


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
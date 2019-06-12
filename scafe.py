import pandas as pd
import numpy as np
from pyts.image import MTF, GASF, GADF
from ae import Encoder, Decoder
import torch
from torch import nn
from torch.optim import Adam
from torch.nn.init import kaiming_normal_
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import os
from sklearn.cluster import AgglomerativeClustering
import sklearn
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from copy import deepcopy

plt.rcParams['figure.figsize'] = (20, 20)

parser = argparse.ArgumentParser()
parser.add_argument('--full', help='set the flag to true if end-to-end is needed', action='store_true')
parser.add_argument('--cluster', help='set the flag to true if only clustering is needed', action='store_true')

# globals
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LATENT_SPACE_DIM = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
BATCH_SIZE = 1024
NUM_CLUSTERS = 2


def read_data(file_path: str) -> np.array:
    """
        Reads the data from disk to a numpy array
        Args:
            file_path: path to the data
        Returns:
            numpy array of the data read from the disk
    """
    return np.array(pd.read_csv(file_path, low_memory=False))


def encode_timeseries(timeseries_tensor: np.array, save: bool = False) -> np.array:
    """
        Encodes the time-series object into images with GASF/GADF/MTF channels
        Args:
            timeseries_tensor: numpy array of clean data
            save: flag for saving
        Returns:
            images: encoded time-series into 3 channel images
    """
    # define the MTF, GASF and GADF transforms
    gasf = GASF(image_size=24)
    gadf = GADF(image_size=24)
    mtf = MTF(image_size=24)

    # transform the time-series
    X_gasf = gasf.fit_transform(timeseries_tensor)
    X_gadf = gadf.fit_transform(timeseries_tensor)
    X_mtf = mtf.fit_transform(timeseries_tensor)

    # get the dimensions of the data
    num_samples, height, width = X_mtf.shape

    # form the images
    images = np.empty(shape=(num_samples, 3, height, width))
    images[:, 0, :, :] = X_gasf[:, :, :]
    images[:, 1, :, :] = X_gadf[:, :, :]
    images[:, 2, :, :] = X_mtf[:, :, :]

    # normalize
    images = (images - np.min(images)) / (np.max(images) - np.min(images))

    # save if needed
    if save:
        np.save(file='./data/GASF_GADF_MTF_images.npy', arr=images)

    return images


def train_autoencoder(images: np.array,
                      batch_size: int,
                      num_epochs: int,
                      learning_rate: float,
                      save: bool,
                      data: np.array):
    """
        Fully trains the autoencoder and writes the metadata to Tensorboard
        Args:
            images: numpy array of the GASF/GADF/MTF images
            batch_size: size of the batch fot the autoencoder
            num_epochs: number of epochs to train the autoencoder for
            learning_rate: learning rate for the weight optimization
            save: flag to save the model to disk
            data: the original clean time-series data -> used for clustering
    """
    def initialize_weights(layer: nn.Module) -> None:
        """
            Initializes the weights for the model
        """
        if type(layer) == nn.Linear or type(layer) == nn.Conv2d or type(layer) == nn.ConvTranspose2d:
            kaiming_normal_(layer.weight)

    def produce_heatmap(encoder: nn.Module, image: np.array):
        """
            Produces the heatmaps, using the modified Grad-CAM algorithm
            Args:
                encoder: the encoder part of the autoencoder that we use to
                         extract the feature maps and the gradients
                image: image we want to get the heatmap of
            Returns:
                Grad-CAM heatmap
        """
        # extract the latent manifold
        manifold = encoder(torch.tensor(image).unsqueeze(dim=0).float())

        # calculate the gradient of the manifold with respect to the parameters of the encoder
        manifold.backward(torch.ones_like(manifold))

        # extract the gradient of the latent manifold with respect ; .....
        gradients = encoder.get_activations_gradient()

        # pool the gradients
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        # get the activations of the last convolutional layer
        activations = encoder.get_activations(torch.tensor(image).unsqueeze(dim=0).float()).detach()

        # weight the channels by corresponding gradients
        for i in range(128):
            activations[:, i, :, :] *= pooled_gradients[i]

        # average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()

        # relu on top of the heatmap
        heatmap = np.maximum(heatmap, 0)

        # normalize the heatmap
        heatmap /= torch.max(heatmap)

        return heatmap.squeeze().numpy()

    # - - - LEARN THE MANIFOLD - - -
    # define the dataset
    dataset = TensorDataset(torch.tensor(images))

    # define the dataloader to feed the data
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=False,
                            num_workers=8 if DEVICE == 'cuda' else 0,
                            pin_memory=True if DEVICE == 'cuda' else False)

    # define the objective
    criterion = nn.BCEWithLogitsLoss().to(DEVICE)

    # init the encoder and decoder
    encoder = Encoder(latent_space_dim=LATENT_SPACE_DIM).to(DEVICE)
    encoder.apply(initialize_weights)
    encoder = nn.DataParallel(encoder) if torch.cuda.device_count() > 1 else encoder

    decoder = Decoder(latent_space_dim=LATENT_SPACE_DIM).to(DEVICE)
    decoder.apply(initialize_weights)
    decoder = nn.DataParallel(decoder) if torch.cuda.device_count() > 1 else decoder

    # define the adam optimizer
    adam = Adam(params=list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)

    # open the tensorboard writer
    writer = SummaryWriter()

    print("Training the autoencoder...")

    # train the model
    for epoch in tqdm(range(1, num_epochs+1)):
        for batch, (imgs) in enumerate(dataloader):

            # put the images onto the device
            imgs = imgs[0].float().to(DEVICE)

            # zero the gradient
            adam.zero_grad()

            # encode the images
            encoded = encoder(imgs)

            # decode the images
            decoded = decoder(encoded)

            # calculate the loss
            loss = criterion(decoded, imgs)

            # backpropagation
            loss.backward()

            # update the weights
            adam.step()

        # write the losses into the tensorboard
        writer.add_scalar(tag='loss', scalar_value=loss.item(), global_step=epoch)
        writer.close()

    # save the model
    if save:
        print('Saving the model...')
        torch.save(obj={'encoder': encoder.state_dict(),
                        'decoder': decoder.state_dict(),
                        'optimizer': adam.state_dict()}, f='./model/scafe_autoencoder.pt')

    # put the model onto the host and set the evaluation mode
    encoder = encoder.module.eval().to(torch.device('cpu'))

    # extract the latent manifold
    manifold = encoder(torch.tensor(images).float()).detach().numpy()

    # save the manifold
    np.save(file='./data/data_manifold.npy', arr=manifold)

    # - - - CLUSTER - - -
    # cluster the manifold
    clusterer = cluster_manifold(data=data)

    # get the indices of the first cluster
    cluster_indices = np.argwhere(clusterer.labels_ == 0).squeeze()

    print('Running the visualizations and saving the plots to disk...')

    # visualize the heatmaps
    fig, axes = plt.subplots(nrows=1, ncols=5)
    fig.set_figwidth(50)
    fig.set_figheight(50)
    for col in range(5):
        heatmap = produce_heatmap(encoder=encoder, image=images[cluster_indices][:5][col])
        heatmap = cv2.resize(heatmap, (24, 24))
        heatmap = np.uint8(255 * heatmap)
        axes[col].imshow(heatmap)
        axes[col].axis('off')
    plt.savefig('./plots/heatmaps.png')

    # visualize the features
    for threshold in [0.01, 0.1, 0.2]:
        fig, axes = plt.subplots(nrows=1, ncols=5)
        fig.suptitle(f'Threshold: {threshold}', fontsize=30)
        fig.set_figwidth(40)
        for col in range(5):
            heatmap = produce_heatmap(encoder=encoder, image=images[cluster_indices][:5][col])
            heatmap = cv2.resize(heatmap, (24, 24))
            heatmap = np.uint8(255 * heatmap)
            gasf_channel = images[cluster_indices][:5][col][0]  # restoration has to come from the GASF
            res = cv2.bitwise_and(gasf_channel, gasf_channel, mask=heatmap)
            d = np.diag(res)
            ddd = np.arccos((d / 2))
            dddd = np.cos(ddd)
            proxy = deepcopy(data[cluster_indices][:5][col])
            proxy[dddd < threshold] = None
            axes[col].plot(data[cluster_indices][:5][col], linewidth=5)
            axes[col].plot(proxy, linewidth=8)
            axes[col].grid(linestyle='--')
            axes[col].set_xlabel('Time', fontsize=30)
            axes[col].tick_params(labelsize=20)
        plt.savefig(f'./plots/projected_features_threshold_{threshold}.png')

    with torch.no_grad():

        # extract the features
        features = encoder.extract_features(torch.tensor(images).float()).numpy()

        # save the features
        np.save(file='./data/data_features.npy', arr=features)


def cluster_manifold(data: np.array) -> sklearn.cluster:
    """
        Creates a partition over the latent manifold
        Args:
            data: the cleaned array of the original time-series data. Used to draw the centroids of the clusters
        Returns:
            agg: the agglomerative clustering scikit-learn model
    """
    def draw_centroids(n_clusters: int, data: np.array, cluster_alg: sklearn.cluster):
        """
            Draws the centroids of the data, given the current trained clustering model.
            Saves the centroids plot to disk
        """
        labels = list()
        for cluster_label in range(n_clusters):
            plt.plot(data[cluster_alg.labels_ == cluster_label].mean(axis=0), linewidth=6)
            labels.append(cluster_label)
        plt.title("Cluster centroids", fontsize=30)
        plt.xlabel("Time Step", fontsize=20)
        plt.xticks(fontsize=20)
        plt.grid(linestyle='--')
        plt.legend(labels=[f'Cluster {label}' for label in labels], prop={'size': 20})
        plt.savefig('./plots/centroids.png')
        plt.clf()

    if not os.path.exists('./data/data_manifold.npy'):
        raise ValueError('Can\'t find data_manifold.npy.'
                         'Run `python scafe.py --train` first.')

    # read the manifold
    manifold = np.load('./data/data_manifold.npy')

    # define the clustering routine
    agg = AgglomerativeClustering(n_clusters=NUM_CLUSTERS,
                                  linkage='ward',
                                  compute_full_tree=True)

    print('Clustering the manifold...')

    # cluster the manifold
    agg.fit(manifold)

    print('Saving the centroids plot to disk...')

    # draw the centroids
    draw_centroids(n_clusters=NUM_CLUSTERS, data=data, cluster_alg=agg)

    return agg


def main(full: bool, cluster: bool):

    # clean_df = read_data('./clean_data.csv')
    clean_data = np.load('./ts_clean.npy')

    if not os.path.exists('./data/GASF_GADF_MTF_images.npy'):
        images = encode_timeseries(clean_data, save=True)
    else:
        images = np.load('./data/GASF_GADF_MTF_images.npy')

    # run the full pipeline
    if full:
        train_autoencoder(images=images,
                          batch_size=BATCH_SIZE,
                          num_epochs=NUM_EPOCHS,
                          learning_rate=LEARNING_RATE,
                          save=True,
                          data=clean_data)

    # run only clustering
    if cluster:
        cluster_manifold(data=clean_data)


if __name__ == '__main__':

    args = parser.parse_args()
    full = args.full
    cluster = args.cluster

    main(full=full, cluster=cluster)

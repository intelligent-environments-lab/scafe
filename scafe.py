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
from tqdm import tqdm


# globals
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LATENT_SPACE_DIM = 128


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
        np.save(file='./GASF_GADF_MTF_images.npy', arr=images)

    return images


def train_autoencoder(images: np.array, batch_size: int, num_epochs: int, learning_rate: float):
    """
        Fully trains the autoencoder and writes the metadata to Tensorboard
        Args:
            images: numpy array of the GASF/GADF/MTF images
            batch_size: size of the batch fot the autoencoder
            num_epochs: number of epochs to train the autoencoder for
            learning_rate: learning rate for the weight optimization
    """
    def initialize_weights(layer: nn.Module):
        """
            Initializes the weights for the model
        """
        if type(layer) == nn.Linear or type(layer) == nn.Conv2d or type(layer) == nn.ConvTranspose2d:
            kaiming_normal_(layer.weight)

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


def main():

    # clean_df = read_data('./clean_data.csv')
    clean_data = np.load('./ts_clean.npy')

    if not os.path.exists('./GASF_GADF_MTF_images.npy'):
        images = encode_timeseries(clean_data, save=True)
    else:
        images = np.load('./GASF_GADF_MTF_images.npy')

    train_autoencoder(images=images, batch_size=128, num_epochs=10, learning_rate=0.001)


if __name__ == '__main__':
    main()

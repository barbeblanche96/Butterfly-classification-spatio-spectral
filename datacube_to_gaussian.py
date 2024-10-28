#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 10:24:50 2024

@author: lisic
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from matplotlib.colors import ListedColormap
import random

# Compute Gaussian distribution parameters from data
def compute_gaussian(data):
    gmm = GaussianMixture(n_components=1)
    gmm.fit(data.reshape(-1, 1))  # Fit a GMM with a single component
    mean_value = gmm.means_[0][0]  # Mean of the fitted Gaussian
    std_dev = np.sqrt(gmm.covariances_[0][0])  # Standard deviation of the Gaussian
    return [mean_value, std_dev[0]]

# Check if the first three rows of the matrix are all zeros
def has_zero_rows(matrix):
    return np.all(matrix[:3, :] == 0)

# Read an HDR metadata file to extract image dimensions (width, height, number of bands)
def lire_fichier_hdr(filename):
    with open(filename, 'r') as fichier:
        lignes = fichier.readlines()

    # Loop through lines to find the image dimensions
    width, heigth, bands = None, None, None
    for ligne in lignes:
        if ligne.startswith('samples'):
            width = int(ligne.split()[-1])
        elif ligne.startswith('lines'):
            heigth = int(ligne.split()[-1])
        elif ligne.startswith('bands'):
            bands = int(ligne.split()[-1])
    
    # Raise an error if any dimension is missing
    if width is None or heigth is None or bands is None:
        raise ValueError("Unable to find image dimensions in the HDR file.")

    return width, heigth, bands

# Set species with options HM, DC, AO or AE.
species = 'HM'

cube_path_1 = f"./datacube/{species}/cube 1/"
cube_path_2 = f"./datacube/{species}/cube 2/"

# Map species to the corresponding datacube and mask files
cube_files_1 = {"HM": "espece1_acq4.dat", "DC": "espece4_acq4.dat", "AO": "espece7_acq10.dat", "AE": "espece10_acq1.dat"}
mask_cube_files_1 = {"HM": "espece1_acq4_mask.npy", "DC": "espece4_acq4_mask.npy", "AO": "espece7_acq10_mask.npy", "AE": "espece10_acq1_mask.npy"}

cube_files_2 = {"HM": "espece1_acq3.dat", "DC": "espece4_acq6.dat", "AO": "espece7_acq11.dat", "AE": "espece10_acq4.dat"}
mask_cube_files_2 = {"HM": "espece1_acq3_mask.npy", "DC": "espece4_acq6_mask.npy", "AO": "espece7_acq11_mask.npy", "AE": "espece10_acq4_mask.npy"}

cluster_nb = {"HM": 3, "DC": 4, "AO": 4, "AE": 3}

# Load the first datacube
dat_file_1 = cube_files_1[species]
width_1, heigth_1, bands_1 = lire_fichier_hdr(cube_path_1 + dat_file_1 + '.hdr')
cube_1 = np.fromfile(cube_path_1 + dat_file_1)
cube_reflectance_1 = cube_1.reshape((bands_1, heigth_1, width_1))

# Load the second datacube
dat_file_2 = cube_files_2[species]
width_2, heigth_2, bands_2 = lire_fichier_hdr(cube_path_2 + dat_file_2 + '.hdr')
cube_2 = np.fromfile(cube_path_2 + dat_file_2)
cube_reflectance_2 = cube_2.reshape((bands_2, heigth_2, width_2))

# Load and apply butterfly masks
butterfly_mask_1 = np.load(cube_path_1 + mask_cube_files_1[species])
butterfly_mask_2 = np.load(cube_path_2 + mask_cube_files_2[species])

# Apply masks to each datacube to remove background
cube_propre_1 = cube_reflectance_1 * butterfly_mask_1
cube_propre_2 = cube_reflectance_2 * butterfly_mask_2

rgb_image_1 = cv2.merge((cube_propre_1[78, :, :], cube_propre_1[29, :, :], cube_propre_1[3, :, :]))
plt.imshow(rgb_image_1)
plt.title(f"{species}: Cube 1 RGB image")
plt.show()

rgb_image_2 = cv2.merge((cube_propre_2[78, :, :], cube_propre_2[29, :, :], cube_propre_2[3, :, :]))
plt.imshow(rgb_image_2)
plt.title(f"{species}: Cube 2 RGB image")
plt.show()

# Retrieve all spectra from each datacube and combine them
spectres_1 = cube_propre_1.reshape((cube_propre_1.shape[0], cube_propre_1.shape[1] * cube_propre_1.shape[2])).T
spectres_2 = cube_propre_2.reshape((cube_propre_2.shape[0], cube_propre_2.shape[1] * cube_propre_2.shape[2])).T

# Combine spectra from both cubes
spectres_combined = np.vstack((spectres_1, spectres_2))

# Define the number of clusters and add one cluster for the background
n_clusters = cluster_nb[species] + 1

# Initialize K-means and perform clustering on the combined spectra
model = KMeans(n_clusters=n_clusters, random_state=42)
labels_combined = model.fit_predict(spectres_combined)

# Separate the labels back for each cube
labels_1 = labels_combined[:spectres_1.shape[0]].reshape((heigth_1, width_1))
labels_2 = labels_combined[spectres_1.shape[0]:].reshape((heigth_2, width_2))

# Define color map for clusters
cmap = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF']
cmap_clusters = random.sample(cmap, k=n_clusters)
cmap_clusters = ListedColormap(cmap_clusters)

# Display the clustering results for each cube
plt.figure(figsize=(12, 6))

# Display Cube 1 with clustering labels
plt.subplot(1, 2, 1)
plt.imshow(cube_propre_1[84], cmap='gray')  # Band example for visualization
plt.imshow(labels_1, cmap=cmap_clusters, alpha=0.5)
plt.colorbar(ticks=range(n_clusters), label='Clusters')
plt.title(f"{species}: Cube 1 with Clustering Labels")

# Display Cube 2 with clustering labels
plt.subplot(1, 2, 2)
plt.imshow(cube_propre_2[84], cmap='gray')  # Band example for visualization
plt.imshow(labels_2, cmap=cmap_clusters, alpha=0.5)
plt.colorbar(ticks=range(n_clusters), label='Clusters')
plt.title(f"{species}: Cube 2 with Clustering Labels")

plt.show()

# Create matrices for mean and standard deviation by cluster and spectral layer
cluster_matrices = [spectres_combined[model.labels_ == i] for i in range(model.n_clusters)]
distribution_mean = np.zeros((n_clusters, spectres_combined.shape[1]), dtype='float64')
distribution_std = np.zeros((n_clusters, spectres_combined.shape[1]), dtype='float64')

# Find the background cluster by checking for zero rows
background_index_cluster = next((i for i, matrix in enumerate(cluster_matrices) if has_zero_rows(matrix)), None)

# Compute Gaussian distribution parameters for each cluster and spectral band
for i in range(n_clusters):
    if i != background_index_cluster:
        group = cluster_matrices[i]
        for j in range(spectres_combined.shape[1]):
            print("Compute Gaussian for cluster:", i, "band:", j)
            data = group[:, j]
            tab = compute_gaussian(data)
            distribution_mean[i, j] = tab[0]
            distribution_std[i, j] = tab[1]


#Save obtained distributions
#np.save(cube_path+'espece1_acq4_distribution_mean.npy', distribution_mean)
#np.save(cube_path+'espece1_acq4_distribution_std.npy', distribution_std)
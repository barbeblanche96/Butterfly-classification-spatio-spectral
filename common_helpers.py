#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 20:09:02 2024

@author: lisic
"""

import numpy as np
from PIL import Image
from scipy.stats import norm


# Load gaussian parameters of species HM
distribution_mean_1 = np.load('./gaussian_parameters/HM/espece1_acq4_distribution_mean_bis.npy')
distribution_std_1 = np.load('./gaussian_parameters/HM/espece1_acq4_distribution_std_bis.npy')

# Load gaussian parameters of species DC
distribution_mean_4 = np.load('./gaussian_parameters/DC/espece4_acq4_distribution_mean.npy')
distribution_std_4 = np.load('./gaussian_parameters/DC/espece4_acq4_distribution_std.npy')


# Load gaussian parameters of species AO
distribution_mean_7 = np.load('./gaussian_parameters/AO/espece7_acq10_distribution_mean.npy')
distribution_std_7 = np.load('./gaussian_parameters/AO/espece7_acq10_distribution_std.npy')


# Load gaussian parameters of species AE
distribution_mean_10 = np.load('./gaussian_parameters/AE/espece10_acq1_distribution_mean_bis.npy')
distribution_std_10 = np.load('./gaussian_parameters/AE/espece10_acq1_distribution_std_bis.npy')


# species and spectralon part
species = ['species1', 'species7', 'species10', 'species4']
spectralon1 = [250, 800]
spectralon4 = [[250, 800], [1500, 2000]]
spectralon7 = [200, 500]
spectralon10 = [100, 450]
spectralon12 = [1300, 1750]


###### 1 sample

#Species DC
images_sp4 = [[['nir_000226.bmp'], ['nir_000234.bmp'], ['nir_000243.bmp'], ['nir_000254.bmp'], ['nir_000260.bmp'], ['nir_000272.bmp'], ['nir_000284.bmp'], ['nir_000320.bmp'], ['vis_000384.bmp'], ['vis_000391.bmp'], ['vis_000396.bmp'], ['vis_000402.bmp'], ['vis_000408.bmp'], ['vis_000410.bmp'], ['vis_000414.bmp'], ['vis_000417.bmp']], [['vis_000208.bmp'], ['vis_000217.bmp'], ['vis_000227.bmp'], ['vis_000231.bmp'], ['vis_000235.bmp'], ['vis_000240.bmp'], ['vis_000246.bmp'], ['vis_000251.bmp'], ['vis_000257.bmp'], ['vis_000265.bmp'], ['vis_000275.bmp'], ['vis_000279.bmp'], ['nir_000309.bmp'], ['nir_000330.bmp'], ['nir_000353.bmp'], ['nir_000367.bmp'], ['nir_000378.bmp'], ['nir_000383.bmp'], ['nir_000393.bmp'], ['nir_000413.bmp'], ['nir_000420.bmp'], ['nir_000428.bmp'], ['nir_000440.bmp'], ['nir_000451.bmp']]]
#Species AO
images_sp7 = [['nir_000321.bmp'], ['nir_000338.bmp'], ['nir_000347.bmp'], ['nir_000350.bmp'], ['nir_000357.bmp'], ['nir_000362.bmp'], ['nir_000363.bmp'], ['nir_000374.bmp'], ['nir_000375.bmp'], ['nir_000379.bmp'], ['nir_000390.bmp'], ['nir_000391.bmp'], ['nir_000396.bmp'], ['nir_000405.bmp'], ['nir_000408.bmp'], ['nir_000445.bmp'], ['nir_000450.bmp'], ['nir_000462.bmp'], ['nir_000465.bmp'], ['nir_000478.bmp'], ['vis_000500.bmp'], ['vis_000530.bmp'], ['vis_000536.bmp'], ['vis_000540.bmp'], ['vis_000544.bmp'], ['vis_000546.bmp'], ['vis_000547.bmp'], ['vis_000552.bmp'], ['vis_000554.bmp'], ['vis_000555.bmp'], ['vis_000561.bmp'], ['vis_000562.bmp'], ['vis_000565.bmp'], ['vis_000570.bmp'], ['vis_000573.bmp'], ['vis_000578.bmp'], ['vis_000596.bmp'], ['vis_000598.bmp'], ['vis_000610.bmp'], ['vis_000620.bmp']]
#Species HM
images_sp1 = [['nir_000364.bmp'], ['nir_000370.bmp'], ['nir_000380.bmp'], ['nir_000386.bmp'], ['nir_000391.bmp'], ['nir_000398.bmp'], ['nir_000405.bmp'], ['nir_000413.bmp'], ['nir_000420.bmp'], ['nir_000430.bmp'], ['nir_000440.bmp'], ['nir_000447.bmp'], ['nir_000455.bmp'], ['nir_000463.bmp'], ['nir_000470.bmp'], ['nir_000484.bmp'], ['nir_000491.bmp'], ['nir_000503.bmp'], ['nir_000510.bmp'], ['nir_000526.bmp'], ['vis_000609.bmp'], ['vis_000612.bmp'], ['vis_000635.bmp'], ['vis_000640.bmp'], ['vis_000645.bmp'], ['vis_000650.bmp'], ['vis_000656.bmp'], ['vis_000660.bmp'], ['vis_000663.bmp'], ['vis_000666.bmp'], ['vis_000669.bmp'], ['vis_000672.bmp'], ['vis_000675.bmp'], ['vis_000678.bmp'], ['vis_000688.bmp'], ['vis_000692.bmp'], ['vis_000695.bmp'], ['vis_000697.bmp'], ['vis_000702.bmp'], ['vis_000705.bmp']]
#Species AE
images_sp10 = [['nir_000247.bmp'], ['nir_000278.bmp'], ['nir_000284.bmp'], ['nir_000300.bmp'], ['nir_000320.bmp'], ['nir_000328.bmp'], ['nir_000354.bmp'], ['nir_000365.bmp'], ['nir_000380.bmp'], ['nir_000442.bmp'], ['nir_000447.bmp'], ['vis_000453.bmp'], ['vis_000460.bmp'], ['vis_000466.bmp'], ['vis_000467.bmp'], ['nir_000476.bmp'], ['vis_000480.bmp'], ['vis_000488.bmp'], ['nir_000489.bmp'], ['nir_000490.bmp'], ['nir_000468.bmp'], ['vis_000492.bmp'], ['vis_000497.bmp'], ['nir_000498.bmp'], ['vis_000500.bmp'], ['nir_000507.bmp'], ['nir_000511.bmp'], ['nir_000514.bmp'], ['nir_000526.bmp'], ['vis_000612.bmp'], ['vis_000613.bmp'], ['vis_000618.bmp'], ['vis_000625.bmp'], ['vis_000628.bmp'], ['vis_000629.bmp'], ['vis_000633.bmp'], ['vis_000640.bmp'], ['vis_000645.bmp'], ['vis_000648.bmp'], ['vis_000656.bmp']]


noise_level_in_db = [10, 12, 14, 16, 18, 20, 22, 24, 26, 30, 32, 34, 36, 38, 40]


# Load a raw image

def remove_blank_area(image):
    # Remove the dead band (120 rows) from the image
    arr = np.delete(image, np.s_[324:444], axis=0)
    
    # Remove the first 4 rows and the last 4 rows
    final_image = arr[4:-4, :]
    
    return final_image

def open_image(path):
    # Open the image from the given path and remove blank areas
    raw_image = Image.open(path)
    return remove_blank_area(np.asarray(raw_image, dtype="uint8"))


# Normalize the raw image

def get_images_by_band(image):
    
    # Number of rows and columns in the image
    n_rows, n_cols = image.shape
    
    n_bands = 5  # Number of bands to split the image into
    band_height = n_rows // n_bands  # Height of each band
    images_by_band = []
    
    # Split the image into bands to process
    for i in range(band_height):
        
        # Index of the first row of the band
        start_row = i * n_bands
        
        # Index of the last row of the band
        end_row = start_row + n_bands
        
        # Extract the band
        band = image[start_row:end_row, :]
        
        images_by_band.append(band)
        
    return images_by_band


def normalize_min_max_by_bands(image):

    images_per_band = get_images_by_band(image)

    rows, cols = image.shape
    
    # Initialize an all-black image
    final_image = np.zeros((rows, cols), dtype='uint8')
    
    band = 1
    
    # Normalize each band to [0, 255]
    for image_band in images_per_band:
        max_pix = np.amax(image_band)  # Find maximum pixel value in the band
        im_norm = image_band / max_pix  # Normalize the image
        # Fill the corresponding region in the final image
        final_image[(band - 1) * 5 : ((band - 1) * 5) + 5] = im_norm * 255
        band += 1
        
    # Adjust brightness using gamma correction
    gamma = 0.7
    gamma_corrected_image = np.power(final_image / 255.0, gamma) * 255.0
    
    return gamma_corrected_image


# Estimate pixel reflectance

def reflectance_image(image, white_begin, white_end):

    images_per_band = get_images_by_band(image)

    rows, cols = image.shape
    
    # Initialize an all-black image
    final_image = np.zeros((rows, cols), dtype='float64')
    
    band = 1
    
    # Normalize reflectance by band using white reference
    for image_band in images_per_band:
        # Get top 50 brightest values in the white region
        white_values = np.sort(np.partition(image_band[:, white_begin:white_end], image_band[:, white_begin:white_end].size - 50, axis=None)[-50:])[::-1]
        im_norm = image_band / np.mean(white_values)  # Normalize by the mean of the brightest white values
        final_image[(band - 1) * 5 : ((band - 1) * 5) + 5] = im_norm
        band += 1
    
    return final_image


# Create a dictionary of reflectances by bands

def dict_reflectance_by_bands(mask, image_reflectance, min_pixels_by_bands=50):
    mask_by_band = get_images_by_band(mask)
    image_reflectance_by_band = get_images_by_band(image_reflectance)
    refl_by_bands = {}
    # Process each band
    for i in range(len(mask_by_band)):
        # Count the number of pixels in the mask
        nb_un = np.count_nonzero(mask_by_band[i] == 1)
        if nb_un >= min_pixels_by_bands:
            # Get reflectance values in the masked region
            reflect_values = image_reflectance_by_band[i][mask_by_band[i] == 1]
            refl_by_bands[str(i)] = reflect_values
    return refl_by_bands


# Calculate Signal-to-Noise Ratio (SNR)

def calculer_snr(signal, signal_bruit):
    # Calculate signal energy
    energie_signal = np.sum(signal**2)
    
    # Calculate noise energy
    energie_bruit = np.sum((signal - signal_bruit)**2)
    
    # Calculate SNR in decibels
    snr = 10 * np.log10(energie_signal / energie_bruit)
    
    return snr


# Create a dictionary of noisy reflectance values by band

def dict_reflectance_noise_by_bands(mask, image_reflectance, elt, min_pixels_by_bands=50):
    refl_by_bands = dict_reflectance_by_bands(mask, image_reflectance, min_pixels_by_bands)
    refl_noise_by_bands = {}
    signal = np.float64([])
    # Combine reflectance values into a single signal array
    for key, value in refl_by_bands.items():
        signal = np.append(signal, value)
    mean = 0
    # Calculate noise standard deviation based on SNR
    std = np.mean(signal) * 10**(-(elt/20))
    noise = np.random.normal(mean, std, signal.shape)
    signal_noise = signal + noise
    print('sigma : ' + str(std), ' snr : ' + str(calculer_snr(signal, signal_noise)) + ' dB')
    # Add noise to each band's reflectance values
    for key, value in refl_by_bands.items():
        refl_noise_by_bands[key] = signal_noise[:len(value)]
        signal_noise = signal_noise[len(value):]
    return refl_noise_by_bands



# Calculate Z-scores for species classification

def zscore_by_specie(distribution_mean, distribution_std, refl_by_bands, append_mode='append'):
    scores = []
    # Compute Z-scores for each band
    for key, value in refl_by_bands.items():
        result_band = np.abs(distribution_mean[:, int(key)][distribution_mean[:, int(key)] != 0] - value[:, np.newaxis])
        std = distribution_std[:, int(key)][distribution_std[:, int(key)] != 0]
        resultat = (result_band / std)
        minimum_par_ligne = np.min(resultat, axis=1)
        # Append or extend the scores depending on mode
        if append_mode == 'append':
            scores.append(minimum_par_ligne)
        else: 
            scores.extend(minimum_par_ligne)
    return scores


# Compute Probability Density Function (PDF) for species classification

def pdf_by_specie(distribution_mean, distribution_std, refl_by_bands, pi):
    scores = []
    # Compute the PDF for each band
    for key, value in refl_by_bands.items():
        probabilites_band = pi * norm.pdf(value[:, np.newaxis], distribution_mean[:, int(key)][distribution_mean[:, int(key)] != 0], distribution_std[:, int(key)][distribution_std[:, int(key)] != 0])
        scores.append(probabilites_band)
    return scores


# PDF with voting mechanism for species classification

def pdf_with_vote_by_specie(distribution_mean, distribution_std, refl_by_bands, pi, append_mode='append'):
    scores = []
    # Compute the PDF and apply voting for each band
    for key, value in refl_by_bands.items():
        probabilites_band = pi * norm.pdf(value[:, np.newaxis], distribution_mean[:, int(key)][distribution_mean[:, int(key)] != 0], distribution_std[:, int(key)][distribution_std[:, int(key)] != 0])
        maximum_par_ligne = np.max(probabilites_band, axis=1)
        if append_mode == 'append':
            scores.append(maximum_par_ligne)
        else:
            scores.extend(maximum_par_ligne)
    return scores



# Aggregate scores for multiple rows

def scores_many_row(rows, winner_value='max'):
    
    final_score = [0, 0, 0, 0]  # Initialize scores for species
    
    images = ''
    domains = ''
    
    # Sum scores for all rows
    for row in rows:
        final_score[0] += row[3]
        final_score[1] += row[4]
        final_score[2] += row[5]
        final_score[3] += row[6]
        images = row[2] + ', '
        domains = row[0] + ', '
    
    # Determine the species with the highest score
    if winner_value == 'max':
        winner_specie = np.argmax(final_score)
    else:
        winner_specie = np.argmin(final_score)
        
    winner = None
    
    # Map species index to species name
    if winner_specie == 0:
        winner = 'species1'
    elif winner_specie == 1:
        winner = 'species4'
    elif winner_specie == 2:
        winner = 'species7'
    elif winner_specie == 3:
        winner = 'species10'
        
    final_row = [domains, rows[0][1], images, final_score[0], final_score[1], final_score[2], final_score[3], winner]
    
    return final_row
        
        

def compute_perf_multi(t, performances):
    general = {'total': 0, 'correct': 0, 'incorrect': 0, 'accurracy': 0}
    for p in performances:
        general['total'] = general['total'] + 1
        if p[1] == p[-1]:
            general['correct'] = general['correct'] + 1
        else:
            general['incorrect'] = general['incorrect'] + 1
    general['accurracy'] = general['correct']/general['total']
    return general



def compute_perf(t, performances):
    general = {'total': 0, 'correct': 0, 'incorrect': 0, 'accurracy': 0}
    vis = {'total': 0, 'correct': 0, 'incorrect': 0, 'accurracy': 0}
    nir = {'total': 0, 'correct': 0, 'incorrect': 0, 'accurracy': 0}
    for p in performances:
        general['total'] = general['total'] + 1
        if p[0] =='nir, ':
            vis['total'] = vis['total'] + 1
            if p[1] == p[-1]:
                vis['correct'] = vis['correct'] + 1
                general['correct'] = general['correct'] + 1
            else:
                vis['incorrect'] = vis['incorrect'] + 1
                general['incorrect'] = general['incorrect'] + 1
        elif p[0] =='vis, ':
            nir['total'] = nir['total'] + 1
            if p[1] == p[-1]:
                nir['correct'] = nir['correct'] + 1
                general['correct'] = general['correct'] + 1
            else:
                nir['incorrect'] = nir['incorrect'] + 1
                general['incorrect'] = general['incorrect'] + 1
    vis['accurracy'] = vis['correct']/vis['total']
    nir['accurracy'] = nir['correct']/nir['total']
    general['accurracy'] = general['correct']/general['total']
    return [general, vis, nir]


def compute_perf_by_species(performances):
    zones = ["vis", "nir"]
    perf = {'species1' : { 'general': [0, 0, 0, 0], 'nir':[0, 0, 0, 0], 'vis': [0, 0, 0, 0] }, 'species4' : { 'general': [0, 0, 0, 0], 'nir':[0, 0, 0, 0], 'vis': [0, 0, 0, 0] },  'species7' : { 'general': [0, 0, 0, 0], 'nir':[0, 0, 0, 0], 'vis': [0, 0, 0, 0] }, 'species10' : { 'general': [0, 0, 0, 0], 'nir':[0, 0, 0, 0], 'vis': [0, 0, 0, 0] }}
    idx = {'species1' : 0, 'species4' : 1, 'species7' : 2, 'species10' : 3}
    for p in performances:
        zone = (p[0].split(',')[0] == zones[0] and "nir") or "vis"
        perf[p[1]][zone][idx[p[-1]]] = perf[p[1]][zone][idx[p[-1]]] + 1
        perf[p[1]]['general'][idx[p[-1]]] = perf[p[1]]['general'][idx[p[-1]]] + 1
    return perf
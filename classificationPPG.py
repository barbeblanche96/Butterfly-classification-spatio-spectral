#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 12:17:25 2024

@author: lisic
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 13:59:56 2024

@author: lisic
"""

import numpy as np
from common_helpers import distribution_mean_1, distribution_std_1, distribution_mean_4, distribution_std_4, distribution_mean_7, distribution_std_7, distribution_mean_10, distribution_std_10, zscore_by_specie, pdf_by_specie, images_sp1, images_sp4, images_sp7, images_sp10, species, spectralon1, spectralon4, spectralon7, spectralon10, dict_reflectance_by_bands, scores_many_row, reflectance_image, open_image, compute_perf, compute_perf_by_species, compute_perf_multi, dict_reflectance_noise_by_bands



def scores_row(weights, total_pix, domain, image_file, species, refl_by_bands, distribution_mean_1, distribution_std_1, distribution_mean_4, distribution_std_4, distribution_mean_7, distribution_std_7, distribution_mean_10, distribution_std_10):    
    

    # # Z-Score
    
    espece_zs_1 = zscore_by_specie(distribution_mean_1, distribution_std_1, refl_by_bands)
    espece_zs_4 = zscore_by_specie(distribution_mean_4, distribution_std_4, refl_by_bands)
    espece_zs_7 = zscore_by_specie(distribution_mean_7, distribution_std_7, refl_by_bands)
    espece_zs_10 = zscore_by_specie(distribution_mean_10, distribution_std_10, refl_by_bands)
    
    zscore_by_band = []
    
    for idx in range(len(espece_zs_1)):
        band_score = np.vstack((espece_zs_1[idx], espece_zs_4[idx], espece_zs_7[idx], espece_zs_10[idx]))
        zscore_by_band.append(band_score.T)
    
    
    # Global matrice
    zscore_global = np.concatenate(zscore_by_band, axis=0)
    
    dist_species = np.sum(zscore_global, axis=0)

    # # PDF
    
    espece_pdf_1 = pdf_by_specie(distribution_mean_1, distribution_std_1, refl_by_bands, 1/14)
    espece_pdf_4 = pdf_by_specie(distribution_mean_4, distribution_std_4, refl_by_bands, 1/14)
    espece_pdf_7 = pdf_by_specie(distribution_mean_7, distribution_std_7, refl_by_bands, 1/14)
    espece_pdf_10 = pdf_by_specie(distribution_mean_10, distribution_std_10, refl_by_bands, 1/14)
    
    probabilities_by_band = []
    
    norm_proba = None
    for idx in range(len(espece_pdf_1)):
        band_probabilities = np.concatenate((espece_pdf_1[idx], espece_pdf_4[idx], espece_pdf_7[idx], espece_pdf_10[idx]), axis=1)
        norm_proba = band_probabilities / np.sum(band_probabilities, axis=1, keepdims=True)
        species_proba = np.column_stack((np.sum(norm_proba[:, :3], axis=1), np.sum(norm_proba[:, 3:7], axis=1), np.sum(norm_proba[:, 7:11], axis=1), np.sum(norm_proba[:, 11:14], axis=1)))
        probabilities_by_band.append(species_proba)
    
    
    # Global matrice
    probabilities_global = np.concatenate(probabilities_by_band, axis=0)
    
    proba_species = np.sum(np.log10(probabilities_global), axis=0)
    
    
    prob_q = np.abs(proba_species)/np.sum(np.abs(proba_species))
    zs_q = dist_species/np.sum(dist_species)


    final_score1, final_score4, final_score7, final_score10 = scores = (weights[0] * (prob_q)) + (weights[1] * (zs_q))
  
    
    winner_specie = np.argmin(scores)
    #winner_specie =  np.argmax(proba_species)
    
    winner = None
    
    if winner_specie == 0:
        winner = 'species1'
    elif winner_specie == 1:
        winner = 'species4'
    elif winner_specie == 2:
        winner = 'species7'
    elif winner_specie == 3:
        winner = 'species10'
    
    row = [domain, species, image_file, final_score1, final_score4, final_score7, final_score10, winner]
    
    return row
    
alpha = [0.92, 0.84]
performances = []
sequence_size = 1

# No noise added to images
noise_in_db = None

# Added a gaussian noise to obtain a snr of 15 dB
#noise_in_db = 15

for sp in species:
    images = None
    spectralon = None
    if sp == 'species1' :
        images = images_sp1
        spectralon = spectralon1
    elif sp == 'species4' :
        images = images_sp4
        spectralon = spectralon4
    elif sp == 'species7' :
        images = images_sp7
        spectralon = spectralon7
    elif sp == 'species10' :
        images = images_sp10
        spectralon = spectralon10


    folder = './classification_test/'+sp+'/'
    
    if sp == 'species4':
        for idy in range(len(images)):
            for idx in range(len(images[idy])):
                
                row_by_samples = []
                
                for j in range (len(images[idy][idx])):
                
                    domain = images[idy][idx][j].split('_')[0]
                    image_file = images[idy][idx][j].split('_')[1]
                    print("Process image -> ", image_file)
                
                    # #Image brute
                    image_brute = open_image(folder+image_file)
                    
                    # #Mask
                    butterfly_mask = np.load(folder+image_file+'_mask.npy')
                    
                    # Calcul des réflectances
                    image_reflectance = reflectance_image(image_brute, spectralon[idy][0], spectralon[idy][1])
                    
                    
                    if domain == 'vis':
                        t = alpha[0]
                    else:
                        t = alpha[1]
                    
                   
                    if noise_in_db == None :
                        refl_by_bands = dict_reflectance_by_bands(butterfly_mask, image_reflectance, 200)
                    else :
                        refl_by_bands = dict_reflectance_noise_by_bands(butterfly_mask, image_reflectance, noise_in_db, 200)
                    
                    
                    total_pix = sum(len(pixels) for pixels in refl_by_bands.values())

                    row_item = scores_row((t, (1-t)), total_pix, domain, image_file, sp, refl_by_bands, distribution_mean_1, distribution_std_1, distribution_mean_4, distribution_std_4, distribution_mean_7, distribution_std_7, distribution_mean_10, distribution_std_10)
                    
                    row_by_samples.append(row_item)
                    
                
                row = scores_many_row(row_by_samples, "min")
                    
                performances.append(row)
            
    else:
    
        for idx in range(len(images)):
            
            row_by_samples = []
            
            for j in range (len(images[idx])):
            
                sequence_size = len(images[idx])
                domain = images[idx][j].split('_')[0]
                image_file = images[idx][j].split('_')[1]
                print("Process image -> ", image_file)
                
                # #Image brute
                image_brute = open_image(folder+image_file)
                
                # #Mask
                butterfly_mask = np.load(folder+image_file+'_mask.npy')
                
                # Calcul des réflectances
                image_reflectance = reflectance_image(image_brute, spectralon[0], spectralon[1])
            
                if domain == 'vis':
                    t = alpha[0]
                else:
                    t = alpha[1]
                               
                
                if noise_in_db == None :
                    refl_by_bands = dict_reflectance_by_bands(butterfly_mask, image_reflectance, 200)
                else :
                    refl_by_bands = dict_reflectance_noise_by_bands(butterfly_mask, image_reflectance, noise_in_db, 200)

                total_pix = sum(len(pixels) for pixels in refl_by_bands.values())
             
                row_item = scores_row((t, (1-t)), total_pix, domain, image_file, sp, refl_by_bands, distribution_mean_1, distribution_std_1, distribution_mean_4, distribution_std_4, distribution_mean_7, distribution_std_7, distribution_mean_10, distribution_std_10)
             
                row_by_samples.append(row_item)
            
        
            row = scores_many_row(row_by_samples, "min")
            
            performances.append(row)
            
print("sequence size :", sequence_size)
print("-------------Perf general------------") 
global_perf = compute_perf_multi((t, (1-t)), performances)
print('global perf : ' +str(global_perf))
if sequence_size == 1:
    print("\n")
    print("-------------Perf by zone------------")
    [general, vis, nir] = compute_perf((t, (1-t)), performances)
    print('general : ' +str(general), 'vis : '+str(vis), 'nir : '+str(nir))
    print("\n")
    print("-------------Perf by species------------")
    perf_by_species = compute_perf_by_species(performances)
    print("perf by species", perf_by_species)
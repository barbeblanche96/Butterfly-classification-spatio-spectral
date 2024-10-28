#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 13:59:56 2024

@author: lisic
"""

import numpy as np
from common_helpers import distribution_mean_1, distribution_std_1, distribution_mean_4, distribution_std_4, distribution_mean_7, distribution_std_7, distribution_mean_10, distribution_std_10, zscore_by_specie, pdf_with_vote_by_specie, images_sp1, images_sp4, images_sp7, images_sp10, species, spectralon1, spectralon4, spectralon7, spectralon10, dict_reflectance_by_bands, scores_many_row, reflectance_image, open_image, compute_perf, compute_perf_by_species, compute_perf_multi, dict_reflectance_noise_by_bands


def scores_row(wiegths, species, refl_by_bands, distribution_mean_1, distribution_std_1, distribution_mean_4, distribution_std_4, distribution_mean_7, distribution_std_7, distribution_mean_10, distribution_std_10):
# Z-score

    espece_zscore_1 = zscore_by_specie(distribution_mean_1, distribution_std_1, refl_by_bands)
    espece_zscore_4 = zscore_by_specie(distribution_mean_4, distribution_std_4, refl_by_bands)
    espece_zscore_7 = zscore_by_specie(distribution_mean_7, distribution_std_7, refl_by_bands)
    espece_zscore_10 = zscore_by_specie(distribution_mean_10, distribution_std_10, refl_by_bands)
    
# # PDF    
    espece_pdf_1 = pdf_with_vote_by_specie(distribution_mean_1, distribution_std_1, refl_by_bands, 1/14)
    espece_pdf_4 = pdf_with_vote_by_specie(distribution_mean_4, distribution_std_4, refl_by_bands, 1/14)
    espece_pdf_7 = pdf_with_vote_by_specie(distribution_mean_7, distribution_std_7, refl_by_bands, 1/14)
    espece_pdf_10 = pdf_with_vote_by_specie(distribution_mean_10, distribution_std_10, refl_by_bands, 1/14)

    
    spectral_vote = np.array([0, 0, 0, 0])
    
    for k in range (len(espece_pdf_1)):
    
        score_zscore_1 = np.zeros(len(espece_zscore_1[k]))
        score_zscore_4 = np.zeros(len(espece_zscore_4[k]))
        score_zscore_7 = np.zeros(len(espece_zscore_7[k]))
        score_zscore_10 = np.zeros(len(espece_zscore_10[k]))
        
        for j in range (len(espece_zscore_1[k])):
            idx_min = np.argmin(np.array([espece_zscore_1[k][j], espece_zscore_4[k][j], espece_zscore_7[k][j], espece_zscore_10[k][j]]))
            if (idx_min == 0):
                score_zscore_1[j] = 1
            elif (idx_min == 1):
                score_zscore_4[j] = 1
            elif (idx_min == 2):
                score_zscore_7[j] = 1
            elif (idx_min == 3) :
                score_zscore_10[j] = 1
                
        final_score_zscore_1 = np.count_nonzero(score_zscore_1 == 1)
        final_score_zscore_4 = np.count_nonzero(score_zscore_4 == 1)
        final_score_zscore_7 = np.count_nonzero(score_zscore_7 == 1)
        final_score_zscore_10 = np.count_nonzero(score_zscore_10 == 1)
        
        
    
            
        # final_score1 = sum(espece_pdf_1)
        # final_score4 = sum(espece_pdf_4)
        # final_score7 = sum(espece_pdf_7)
        # final_score10 = sum(espece_pdf_10)
        
        score_pdf_1 = np.zeros(len(espece_pdf_1[k]))
        score_pdf_4 = np.zeros(len(espece_pdf_4[k]))
        score_pdf_7 = np.zeros(len(espece_pdf_7[k]))
        score_pdf_10 = np.zeros(len(espece_pdf_10[k]))
        
        for j in range (len(espece_pdf_1)):
            idx_max = np.argmax(np.array([espece_pdf_1[k][j], espece_pdf_4[k][j], espece_pdf_7[k][j], espece_pdf_10[k][j]]))
            if (idx_max == 0):
                score_pdf_1[j] = 1
            elif (idx_max == 1):
                score_pdf_4[j] = 1
            elif (idx_max == 2):
                score_pdf_7[j] = 1
            elif (idx_max == 3) :
                score_pdf_10[j] = 1
                
        
        final_score_pdf_1 = np.count_nonzero(score_pdf_1 == 1)
        final_score_pdf_4 = np.count_nonzero(score_pdf_4 == 1)
        final_score_pdf_7 = np.count_nonzero(score_pdf_7 == 1)
        final_score_pdf_10 = np.count_nonzero(score_pdf_10 == 1)
        
        
        
        # # PDF + Zscore
        
        
        final_score1 = (wiegths[1] * final_score_zscore_1) + (wiegths[0] * final_score_pdf_1)
        final_score4 = (wiegths[1] * final_score_zscore_4) + (wiegths[0] * final_score_pdf_4)
        final_score7 = (wiegths[1] * final_score_zscore_7) + (wiegths[0] * final_score_pdf_7)
        final_score10 = (wiegths[1] * final_score_zscore_10) + (wiegths[0] * final_score_pdf_10)
    

        winner_specie_k =  np.argmax([final_score1, final_score4, final_score7, final_score10])
        
        spectral_vote[winner_specie_k] = spectral_vote[winner_specie_k] + 1 
        
    
    
    winner_specie =  np.argmax(spectral_vote)

    winner = None
    
    if winner_specie == 0:
        winner = 'species1'
    elif winner_specie == 1:
        winner = 'species4'
    elif winner_specie == 2:
        winner = 'species7'
    elif winner_specie == 3:
        winner = 'species10'
    
    row = [domain, species, image_file, spectral_vote[0], spectral_vote[1], spectral_vote[2], spectral_vote[3], winner]
    
    return row



alpha = [0.74, 0.90]
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
                    
                    row_item = scores_row((t, (1-t)), sp, refl_by_bands, distribution_mean_1, distribution_std_1, distribution_mean_4, distribution_std_4, distribution_mean_7, distribution_std_7, distribution_mean_10, distribution_std_10)
                    
                    row_by_samples.append(row_item)
                
                row = scores_many_row(row_by_samples)
                    
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
                
                row_item = scores_row((t, (1-t)), sp, refl_by_bands, distribution_mean_1, distribution_std_1, distribution_mean_4, distribution_std_4, distribution_mean_7, distribution_std_7, distribution_mean_10, distribution_std_10)
                
                row_by_samples.append(row_item)            
        
            row = scores_many_row(row_by_samples)
                
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
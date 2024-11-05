
# Butterfly Recognition With Raw Spatio-spectral Images

This project describes a Novel Statistical Framework for Butterfly Species Recognition Using Raw Spatio-Hyperspectral Images. This project presents all the codes implemented for the experiments conducted in this [paper](). The experiments were carried out on four butterfly species: Hypolimnas Misippus (HM), Danaus Chrysippus (DC), Amauris Ochlea (AO), and Acraea Egina (AE). The project has been conducted under the financial support of International Institute of Tropical Agriculture (IITA) in Benin.

## Dataset
Download the dataset from the following [link](https://zenodo.org/records/14004272) and unzip it in the root directory of the project.

## Repository organization

- `classification_test/` : Contains raw spatio-spectral images related to the testing of classification models.
  - `{species}/` : Contains the raw images for each butterfly `{species}`
  
- `datacube/` : Contains hyperspectral datacubes for each species, used to estimate Gaussian distribution parameters.
  - `{species}/cube {#No}` : Contains the specific datacube for the butterfly `{species}`
      - `{filename}.dat` : Data file containing the primary spectral information of the butterfly `{species}`.
      - `{filename}.dat.hdr` : Header file containing metadata for the corresponding `{filename}.dat` file, including information about dimensions, wavelengths, and other important parameters.
      - `{filename}_mask.npy` : A NumPy array file that likely contains a mask to segment butterfy regions in the hyperspectral datacube.


- `gaussian_parameters/` : Contains files related to Gaussian parameters for data modeling and analysis.
  - `{species}/{filename}_mean.npy` : NumPy array file that stores the mean values of the Gaussian distribution for the `{species}`.
  - `{species}/{filename}_std.npy` : NumPy array file that stores the standard deviation values of the Gaussian distribution for the `{species}`.

## Python Files

- `classificationPPG.py` : Python script for classification using "PPG" method.
  
- `classificationPPS.py` : Python script for classification using "PPS" method.
  
- `classificationVPG.py` : Python script for classification using "VPG" method.
  
- `classificationVPS.py` : Python script for classification using "VPS" method.

- `common_helpers.py` : A script containing common utility functions that are used across different parts of the project.

- `datacube_to_gaussian.py` : Python script to estimate Gaussian parameters from hyperspectral datacube of each species.


## Running python files

Each python file can be run separately, depending on the method to be used or the task to be performed

### Estimation of Gaussian parameters
By running the python file "datacube_to_gaussian.py", you can estimate the parameters for a given species from its corresponding data cube. To do so, locate the line of code below and specify the desired species by choosing one of the following options: 'HM', 'DC', 'AO', or 'AE'.


```python
# Set species with options HM, DC, AO, or AE.
species = 'AE' 
```


### Evaluation using the test dataset
"To obtain the performance results on the test dataset using a specific method, run any of the following Python files: classificationPPG.py, classificationPPS.py, classificationVPG.py, or classificationVPS.py."


To add Gaussian noise to the images to evaluate performance in the presence of noise, find the following line in the Python file corresponding to the method used and specify the desired SNR value in dB :

```python
# No noise added to images.
# You can specify any value in dB
noise_in_db = None
```
## Additional Notes

Aliases have been assigned to each species

- species HM : species 1
- species DC : species 4
- species AO : species 7
- species AE : species 10

# eswremove
Removes electrical standing waves in HIFI spectrometer observations using machine learning. HIFI is a far-infrared spectrometer on board of the Herschel Space Telescope. HIFI's bands that use HEB mixers have strong electrical standing waves (ESWs), a type of instrumental noise that causes ripples in the baseline of a spectrum. This algorithm removes them to obtain spectra with flat baselines.

## overview
The algorithm features:
- a CNN that removes non-baseline features
- a PCA transformation that uses a linear combination of eigenvectors to describe a baseline
- a dense neural network for small corrections

The PCA is fitted to HIFI stability data. Stability data contains spectra without astronomical signals, and which thus consist mostly of ESW noise. After the PCA is fitted to the stability data, its eigenvectors give a good representation of any ESWs in other spectra.

## file structure
- eswremove.py: contains the function definitions used for training and testing purposes
- data: contains the stability data used for training, and CII observations from the GOT C+ survey, either including or excluding the default pipeline method that removes ESWs

## credits
- Author: Roy Bos
- Thesis supervisor: Russell Shipman
- Collaborator: Youngmin Seo

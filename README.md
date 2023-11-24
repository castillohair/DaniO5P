# DaniO5P: Predicting 5'UTR-mediated translation during zebrafish embryogenesis

This repository contains code to train, evaluate, and interpret the Danio Optimus 5-Prime (DaniO5P) model from the publication "[The regulatory landscape of 5′ UTRs in translational control during zebrafish embryogenesis](https://www.biorxiv.org/content/10.1101/2023.11.23.568470v1)".

# Contents

- `00_data`: Data preprocessing. Computes mean ribosome loading (MRL) and differences in total RNA TPMs across timepoints, which are used later for model fitting.
- `01_length_model`: Calculate and evaluate a model on MRL and TPM differences only based on 5'UTR length. Compute predictions and residuals (measurements - predictions) for all MPRA sequences.
- `02_cnn`: Train and evaluate an ensemble of convolutional neural network (CNN) models to predict the residuals of MRL and TPM differences based on sequence.
- `03_full_model_evaluation`: Compute performance metrics on the full DaniO5P (length + CNN) model, which are reported in the manuscript. In addition, notebook `evaluate_full_model.ipynb` computes full ensemble model predictions for all MPRA sequences, and can be used as a general example on how to use the full model.
- `04_interpretation`: Use DeepSHAP (see below) to calculate how each nucleotide in all MPRA sequences contributes to MRL predictions. Generate nucleotide contribution plots.
- `05_filter_motifs`: Extract motifs from the convolutional filters of the CNN models, cluster them, and calculate average motif contribution to MRL and TPM differences at each timepoint.
- `utils`: Supporting code.

More details are available in each of the notebook and code files within each folder.

# Package requirements
All of the code here was run in Python 3.9 with the following package version:
- `matplotlib` 3.5.1
- `numpy` 1.22.1
- `pandas` 1.4.3
- `scipy` 1.7.3
- `seaborn` 0.12.2
- `logomaker` 0.8

Most of the deep learning code here was run in `tensorflow` 2.4. The only exception is `04_interpretation/run_interpretation_in_avg_model.py`, which requires `tensorflow` 1.15 instead.

Other software used includes:
- The DeepSHAP version at https://github.com/kundajelab/shap commit # 29d2ffab405619340419fc848de6b53e2ef0f00c for the code in `04_interpretation`.
- A modified version of `matrix-clustering` at https://github.com/castillohair/matrix-clustering_stand-alone, original from https://github.com/jaimicore/matrix-clustering_stand-alone, for the code in `05_filter_motifs`.

# Other requirements

The following files must be downloaded separately:

- Trained tensorflow models can be downloaded from [TODO: add URL when available]
- Calculated contribution scores for all MPRA sequences can be downloaded from [TODO: add URL when available]

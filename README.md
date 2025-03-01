# Cover Song Similarity Using a Linear Autoencoder
## Table of Contents
- [Cover Song Similarity Using a Linear Autoencoder](#cover-song-similarity-using-a-linear-autoencoder)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Dataset](#dataset)
  - [Experimentation](#experimentation)
  - [Evaluation](#evaluation)
  - [Improvements](#improvements)
## Introduction
This project explores cover song similarity using a linear autoencoder trained on the Da-TACOS dataset. The autoencoder reconstructs audio embeddings while capturing essential similarities across different song versions.

## Dataset
The Da-TACOS (Database of Temporal Alignments of Cover Songs) dataset, developed by the Music Technology Group (MTG), provides a large-scale collection of cover songs with detailed time-aligned annotations. It contains:

Over 16,000 cover song pairs across various genres.
Frame-level alignment annotations, making it ideal for studying musical version identification.
A diverse set of features, including Chroma, HPCP, MFCCs, and Spectrograms.

However, in this project, we only use aggregated metadata rather than raw spectrograms or time-series features.

## Experimentation
Initially, only Chroma and HPCP features were extracted. This decision was based on insights from the paper:
"Audio-based Musical Version Identification: Elements and Challenges"
by Furkan Yesiler, Guillaume Doras, Rachel M. Bittner, Christopher J. Tralie, Joan Serra

This follows the intuition that cover songs usually retain similar harmonic and melodic structures but may differ in timbre, tempo, or instrumentation.

Furthermore, after the first run it became apparent that the embedding separation was not sufficient so a second experiment was run with all 52 features from the dataset.

The Mean Squared Error (MSE) was chosen as the loss function to favor accurate reconstruction of the audio embeddings.

## Evaluation
For the evaluation we check the similarity of embeddings by computing distances between audio performances of the same musical work (in-clique) and performances of different works (cross-clique).

We observe that distributions from both experiments have a significant degree of overlap.

<div style="display: flex; justify-content: center; align-items: center; gap: 10px; max-width: 900px; margin: auto;">
  <figure style="text-align: center; min-width: 300px;">
    <img src="https://github.com/user-attachments/assets/91dd6bde-2fbc-4da3-a935-a49946f30bc4" width="280">
    <figcaption>Distances using chroma and hpcp only.</figcaption>
  </figure>
  <figure style="text-align: center; min-width: 300px;">
    <img src="https://github.com/user-attachments/assets/0fb746d6-d098-46c4-81d2-bd9005825446" width="280">
    <figcaption>Distances using all aggregated features.</figcaption>
  </figure>
</div>


This may be from the following:
 1. Each feature vector is aggregated, i.e. (12,N)->(12,1).
 2. The metric used to build the latent space may be lacking.
 3. A linear autoencoder might be too simple to capture complex musical relationships.

## Improvements
To improve performance, several modifications could be explored:

Hybrid Loss Function:
MSE → Ensures high-fidelity reconstruction.
Contrastive Loss → Encourages the model to learn a more discriminative embedding space, where cover songs are pulled closer together while unrelated songs are pushed apart.

Sequence-Based Features:
Instead of using aggregated metadata, we could incorporate time-series features such as MFCCs or spectrograms.
A 1D Convolutional Neural Network (CNN) could then be used to extract richer temporal dependencies, improving the ability to model local structures in the audio embeddings.

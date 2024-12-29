# Generative Adversarial Networks (GANs) Project

This repository contains the work done for **Data Science Lab Project 2 (IASD)**, in collaboration with **Simon Liétar** and **Ilian Benaïssa-Lejay**. The project explores **Generative Adversarial Networks (GANs)** and their variants, focusing on generating high-quality MNIST-like images.

## 📜 Summary
This project evaluates and improves various GAN architectures, including:
- **Vanilla GAN**: The baseline implementation.
- **WGAN**: Introduces Wasserstein distance for improved stability.
- **WGAN-GP**: Incorporates Gradient Penalty for enhanced training and better results.
- **Rejection Sampling**: Filters generated images based on Inception Scores to improve sample quality.

## 📂 Repository Structure
### Key Files
- **`utils.py`**: Supports the training of Vanilla GAN, WGAN, and WGAN-GP with helper functions for data processing and logging.
- **`generate.py`**: Generates images using trained GAN models.
- **`model.py`**: Implements the GAN architecture and model components.
- **`requirements.txt`**: Required Python packages for the project.
- **`Report.pdf`**: A detailed report of the project, including methodologies, experiments, and results.

### Folders
- **`Vanilla_GAN/`**: Training loop for the baseline GAN model.
- **`WGAN/`**: Training loop for Wasserstein GAN.
- **`WGANGP/`**: Training loop for WGAN with Gradient Penalty.
- **`rejection_sampling/`**: Code and experiments for improving sample quality with rejection sampling.
- **`checkpoints/`**: Saved models for reproducibility.
- **`plots_code/`**: Scripts for visualizing generated samples and performance metrics.
- **`raw/`**: Raw datasets used for training and evaluation.



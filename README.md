# Faithful Interpretability for Acoustic Anomaly Detection

This project investigates how interpretability methods can provide faithful and human-aligned explanations for deep learning acoustic anomaly detection models.

We compare a standard Autoencoder (AE) and a Masked Autoencoder (MAE) trained on real industrial planer sounds (data is available [here](https://github.com/AnthonyDeschenes/PlaningItByEarDataset/tree/main)), and evaluate several post-hoc explanation methods: error maps, saliency maps, Integrated Gradients, SmoothGrad, GradSHAP, and Grad-CAM.

To assess explanation quality, we apply a perturbation-based faithfulness metric and use expert-annotated anomalies for quantitative evaluation.

The localized annotated anomaly data is available [here](https://docs.google.com/spreadsheets/d/1dcYCwxwJPJapTGzIUeMZsNLjyiTSxe55j4NrVN39BYQ/edit?usp=sharing).

# Usage:

 - Install dependencies listed in `requirements.txt`.
 - The notebooks `train-models.ipynb` and `evaluate-models.ipynb` contain the script to train and evaluate different anomaly detection models. The convolutional autoencoder with skip connections and transformer achieves the best baseline performance.
 - In `maskedAE.ipynb`, the best transformer autoencoder model is retrained with masked input regions (MAE training) and compared against the baseline autoencoder using both faithfulness and F-score metrics.
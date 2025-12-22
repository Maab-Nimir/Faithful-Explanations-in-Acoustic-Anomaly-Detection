# Faithful Interpretability for Acoustic Anomaly Detection

This project investigates how interpretability methods can provide faithful and human-aligned explanations for deep learning acoustic anomaly detection models.

We compare a standard Autoencoder (AE) and a Masked Autoencoder (MAE) trained on real industrial planer sounds (data is available [here](https://github.com/AnthonyDeschenes/PlaningItByEarDataset/tree/main)), and evaluate several post-hoc explanation methods: error maps, saliency maps, Integrated Gradients, SmoothGrad, GradSHAP, and Grad-CAM.

To assess explanation quality, we apply a perturbation-based faithfulness metric and use expert-annotated anomalies for quantitative evaluation.

The localized annotated anomaly data is available [here](https://docs.google.com/spreadsheets/d/1dcYCwxwJPJapTGzIUeMZsNLjyiTSxe55j4NrVN39BYQ/edit?usp=sharing).

# Usage:

 - Install dependencies listed in `requirements.txt`.
 - The notebooks `train-models.ipynb` and `evaluate-models.ipynb` contain the script to train and evaluate different anomaly detection models. The convolutional autoencoder with skip connections and transformer achieves the best baseline performance.
 - In `maskedAE.ipynb`, the best transformer autoencoder model is retrained with masked input regions (MAE training) and compared against the baseline autoencoder using both faithfulness and F-score metrics.
 - The notebook `dcase2022.ipynb` presents initial experiments to evaluate the applicability of the proposed interpretability framework on a different dataset. The anomaly detection models are not optimized and are intended for exploratory analysis; performance can be improved with hyperparameter tuning and additional training data.
 - A transformer-based AE architecture was adapted to DCASE inputs, and both AE and MAE models were trained on the development training set for 500 epochs with early stopping (patience = 30). Results on the development evaluation set show comparable detection performance across machine types, with best performance on fan machines and lowest on valve. Mean source/target AUCs are 0.513/0.495 (AE) and 0.509/0.463 (MAE).
 - Using frame-based faithfulness (fidelity) and averaging the faithfulness scores at the highest threshold percentiles (98th–99th), MAE achieves slightly higher faithfulness than AE (0.039 vs. 0.038), indicating more reliable explanations even before model optimization.
## DCASE 2022 Task 2 – Anomaly Detection Results (Preliminary)

### Autoencoder (AE)

| Machine Type | Source AUC | Target AUC | pAUC |
|-------------|------------|------------|------|
| Toy Car     | 0.508 | 0.521 | 0.509 |
| Toy Train  | 0.488 | 0.596 | 0.529 |
| Bearing    | 0.546 | 0.560 | 0.508 |
| Fan        | 0.605 | 0.544 | 0.514 |
| Gearbox   | 0.557 | 0.472 | 0.493 |
| Slider     | 0.516 | 0.436 | 0.505 |
| Valve      | 0.373 | 0.338 | 0.484 |
| **Mean**   | **0.513** | **0.495** | **0.506** |

### Masked Autoencoder (MAE)

| Machine Type | Source AUC | Target AUC | pAUC |
|-------------|------------|------------|------|
| Toy Car     | 0.546 | 0.545 | 0.520 |
| Toy Train  | 0.420 | 0.460 | 0.491 |
| Bearing    | 0.503 | 0.440 | 0.506 |
| Fan        | 0.600 | 0.567 | 0.525 |
| Gearbox   | 0.586 | 0.469 | 0.514 |
| Slider     | 0.554 | 0.428 | 0.527 |
| Valve      | 0.356 | 0.331 | 0.489 |
| **Mean**   | **0.509** | **0.463** | **0.510** |

### Mean faithfulness of the highest threshold percentiles (98th–99th) for each model and each interpretability method

| Model | Error map | Saliency map | IG   | Smooth Grad | Grad CAM | Grad SHAP |
|-------|-----------|--------------|------|-------------|----------|-----------|
| AE    | **0.038**     | 0.036        | 0.027| 0.036       | 0.020    | 0.027     |
| MAE   | **0.039**     | 0.029        | 0.024| 0.030       | 0.017    | 0.024     |


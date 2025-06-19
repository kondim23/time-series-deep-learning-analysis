# Deep Learning for Time Series: Forecasting, Anomaly Detection, and Representation Learning

## Overview
This project presents a modular framework for time series forecasting, anomaly detection, and representation learning using deep learning. The focus is on stock time series, leveraging LSTM and convolutional autoencoder architectures for robust, scalable analysis.

## Objectives
- Predict future values of time series using LSTM networks
- Detect anomalies in time series with LSTM autoencoders
- Learn compact representations of time series using convolutional autoencoders
- Provide reproducible experiments and clear visualizations

## Theoretical Background
### Time Series Analysis
Time series data consists of sequential measurements over time. Accurate analysis is crucial in finance, engineering, and science for forecasting, anomaly detection, and pattern discovery.

### LSTM Networks
Long Short-Term Memory (LSTM) networks are a type of recurrent neural network (RNN) designed to capture long-term dependencies in sequential data. They are effective for time series forecasting due to their ability to model temporal patterns and mitigate vanishing gradient issues.

### Autoencoders
Autoencoders are neural networks trained to reconstruct their input. LSTM autoencoders are used for sequence reconstruction and anomaly detection, while convolutional autoencoders learn compressed representations for clustering and retrieval.

## Methodology
### Data Preparation
- Input: Tab-separated files, each row is a time series
- Preprocessing: Scaling, windowing, and splitting into train/test sets

### Forecasting
- Model: Deep LSTM network (4–8 layers, 64–128 units per layer, dropout 0.2)
- Training: Early stopping, validation loss monitoring
- Evaluation: Forecast plots, train/test loss curves

### Anomaly Detection
- Model: LSTM autoencoder (4–6 layers, 32–64 units, dropout)
- Training: On normal data, validated on mixed data
- Evaluation: MAE threshold for anomaly flagging, anomaly visualization

### Representation Learning
- Model: Convolutional autoencoder (2–3 downsampling/upsampling layers, filter size 16, latent dimension 7–13)
- Training: Windowed time series, batch normalization, dropout
- Evaluation: Compression quality, clustering, and search on learned representations

## Model Selection and Hyperparameters
- Hyperparameters were tuned based on validation loss and visual inspection
- Dropout and early stopping were used to prevent overfitting
- Final models balance accuracy, generalization, and computational efficiency

## Experimental Results
- LSTM forecasting achieved low validation loss and accurate predictions
- LSTM autoencoder detected anomalies with high precision using MAE thresholding
- Convolutional autoencoder produced compact, informative representations for clustering
- Results are visualized with plots for loss, predictions, anomalies, and encoded series

## Discussion
- LSTM models are effective for both forecasting and anomaly detection in time series
- Convolutional autoencoders enable dimensionality reduction and facilitate downstream tasks
- Hyperparameter tuning and regularization are critical for robust performance
- Limitations include sensitivity to data quality and the need for careful threshold selection in anomaly detection

## Conclusions and Future Work
This project demonstrates the effectiveness of deep learning for time series analysis. Future work may include:
- Exploring attention mechanisms or transformer models
- Applying models to other domains (e.g., sensor data, IoT)
- Automating hyperparameter optimization
- Integrating explainability techniques for model interpretation

---

This report summarizes the methodology, experiments, and findings of the project, providing a foundation for further research and practical applications in time series analysis.

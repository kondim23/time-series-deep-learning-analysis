# Deep Learning for Time Series: Forecasting, Anomaly Detection, and Representation Learning

This repository contains the project as originally developed in [time-series-dl.ipynb](https://gist.github.com/kondim23/3a000579d870bd0ca7edc314db58f2fa), now extended and refactored for clarity, reproducibility, and extensibility.

## Problem Statement
Time series analysis is essential in fields like finance, engineering, and science for forecasting, anomaly detection, and pattern discovery. In stock markets, accurate forecasting and anomaly detection drive better investment decisions, risk management, and fraud detection. However, stock time series are high-dimensional, noisy, and non-stationary, making traditional methods insufficient for robust analysis.

## Solution Approach & Key Features
This project addresses the challenges of time series analysis by leveraging deep learning techniques and providing:
- **LSTM Networks** for forecasting and anomaly detection, capturing long-term dependencies and complex temporal patterns.
- **Convolutional Autoencoders** for dimensionality reduction and representation learning, enabling efficient clustering and similarity search in high-dimensional spaces.

## Collaboration
This repository was developed in cooperation with the [kondim/highdim-curve-search-clustering](https://github.com/kondim23/highdim-curve-search-clustering), which was used to verify and validate some of the results produced here, particularly for high-dimensional search and clustering tasks using the compressed time series representations.

## Project Recap
See [doc/report.md](doc/report.md) for detailed methodology, experiments, and results. Highlights include:
- Extensive hyperparameter tuning and validation for all models.
- Visual and quantitative evaluation of forecasting, anomaly detection, and compression.
- External validation of compressed representations for clustering and search.

## Technologies Used

- **Python 3.7+** — Core programming language for all scripts and notebooks
- **TensorFlow & Keras** — Deep learning frameworks for building and training LSTM and convolutional autoencoder models
- **NumPy & pandas** — Data manipulation and preprocessing
- **matplotlib** — Visualization of results and model performance
- **Jupyter Notebook** — Interactive experimentation and documentation

## Project Structure
```
project-algorithms-ml/
├── data/                        # Raw and processed datasets
├── notebooks/                   # Jupyter notebooks for experiments
│   └── project_algorithms.ipynb
├── src/                         # Source code for models and utilities
│   ├── forecasting.py
│   ├── anomaly_detection.py
│   ├── dimensionality_reduction.py
│   ├── utils.py
│   └── dataset_splitter.py
├── doc/                         # Documentation and reports
│   ├── report.md
├── README.md
```

## Installation
- Python 3.7+
- TensorFlow, Keras, NumPy, pandas, matplotlib

## Usage
### Input Format
Tab-separated files, each row is a time series:
```
item_id1 X11 X12 ... X1d
item_idN XN1 XN2 ... XNd
```

### Forecasting
Train and evaluate an LSTM model for time series prediction:
```bash
python3 forecasting.py -d <dataset path> -n <number of time series selected> -t <offline_all|online_all|online_self>
```
- `-t offline_all`: Use a pre-trained model for all time series
- `-t online_all`: Train on all time series during execution
- `-t online_self`: Train per time series during execution

### Anomaly Detection
Detect anomalies in time series using LSTM autoencoders:
```bash
python3 anomaly_detection.py -d <dataset path> -n <number of time series selected> -t <offline_all|online_all> -mae <error value as double>
```
- `-mae`: Mean Absolute Error threshold for anomaly detection

### Dimensionality Reduction & Representation Learning
Learn compact representations with convolutional autoencoders:
```bash
python3 dimensionality_reduction.py -d <dataset> -q <queryset> -od <output_dataset_file> -oq <output_query_file> -t <offline_all|online_all>
```
- Outputs reduced representations in tab-separated files for downstream tasks

### Dataset Splitting
Split a dataset into train/query sets:
```bash
python3 split_dataset.py <dataset_path>
```

## Conclusion
This project demonstrates the effectiveness of deep learning for time series forecasting, anomaly detection, and representation learning. By combining LSTM and convolutional autoencoder architectures, it provides a framework for analyzing complex stock time series data. The codebase is designed for clarity, reproducibility, and future extension to other domains or advanced architectures.

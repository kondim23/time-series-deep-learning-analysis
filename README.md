# Deep Learning for Time Series: Forecasting, Anomaly Detection, and Representation Learning

This repository contains the project as originally developed in [kondim/repository](https://github.com/kondim/repository), now extended and refactored for clarity, reproducibility, and extensibility.

## Problem Statement
Time series analysis is essential in fields like finance, engineering, and science for forecasting, anomaly detection, and pattern discovery. In stock markets, accurate forecasting and anomaly detection drive better investment decisions, risk management, and fraud detection. However, stock time series are high-dimensional, noisy, and non-stationary, making traditional methods insufficient for robust analysis.

## Solution Approach & Key Features
This project addresses the challenges of time series analysis by leveraging deep learning techniques and providing:
- **LSTM Networks** for forecasting and anomaly detection, capturing long-term dependencies and complex temporal patterns.
- **Convolutional Autoencoders** for dimensionality reduction and representation learning, enabling efficient clustering and similarity search in high-dimensional spaces.
- **External Validation:** Compressed time series representations are validated through high-dimensional search and clustering in the [kondim/repository](https://github.com/kondim/repository) project, confirming their practical utility.

## Project Recap
See [doc/report.md](doc/report.md) for detailed methodology, experiments, and results. Highlights include:
- Extensive hyperparameter tuning and validation for all models.
- Visual and quantitative evaluation of forecasting, anomaly detection, and compression.
- External validation of compressed representations for clustering and search.

## Project Structure
```
project-algorithms-ml/
├── data/                        # Raw and processed datasets
├── notebooks/                   # Jupyter notebooks for experiments
│   └── project_algorithms.ipynb
├── src/                         # Source code for models and utilities
│   ├── forecasting.py
│   ├── anomaly_detection.py
│   ├── compression.py
│   ├── common_utils.py
│   └── split_dataset.py
├── results/                     # Output results, figures, and tables
│   ├── figures/
│   ├── tables/
│   └── logs/
├── doc/                         # Documentation and reports
│   ├── report.md
│   └── README.md
├── requirements.txt             # Python dependencies
├── LICENSE
└── .gitignore
```

## Installation
- Python 3.7+
- TensorFlow, Keras, NumPy, pandas, matplotlib (see `requirements.txt`)

Install dependencies:
```bash
pip install -r requirements.txt
```

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
python3 forecast.py -d <dataset path> -n <number of time series selected> -t <offline_all|online_all|online_self>
```
- `-t offline_all`: Use a pre-trained model for all time series
- `-t online_all`: Train on all time series during execution
- `-t online_self`: Train per time series during execution

### Anomaly Detection
Detect anomalies in time series using LSTM autoencoders:
```bash
python3 detect.py -d <dataset path> -n <number of time series selected> -t <offline_all|online_all> -mae <error value as double>
```
- `-mae`: Mean Absolute Error threshold for anomaly detection

### Dimensionality Reduction & Representation Learning
Learn compact representations with convolutional autoencoders:
```bash
python3 reduce.py -d <dataset> -q <queryset> -od <output_dataset_file> -oq <output_query_file> -t <offline_all|online_all>
```
- Outputs reduced representations in tab-separated files for downstream tasks

### Dataset Splitting
Split a dataset into train/test/query sets:
```bash
python3 split_dataset.py <dataset_path>
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

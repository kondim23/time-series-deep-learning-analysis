# Stock Time Series Analysis with Deep Learning

## Overview
This repository provides a professional, production-ready toolkit for advanced time series analysis of stock data using state-of-the-art deep learning techniques. Developed originally as a university assignment, it is now structured for clarity, reproducibility, and extensibility, making it suitable for both academic and industry review.

## What does this repository do?
- **Forecasts stock prices** using multi-layer LSTM neural networks, with extensive hyperparameter tuning and overfitting prevention.
- **Detects anomalies** in stock time series via LSTM autoencoders, with threshold-based detection and visualization.
- **Learns compact representations** of time series using convolutional autoencoders, enabling clustering and similarity search.
- **Compares clustering and search results** before and after dimensionality reduction.
- **Provides modular, well-documented code** and Jupyter notebooks for experimentation and demonstration.

## Features
- Command-line tools for each task:
  - `forecast.py`: Stock price forecasting
  - `detect.py`: Anomaly detection
  - `reduce.py`: Dimensionality reduction and representation learning
- Flexible training modes: offline (pre-trained), online (train on all series), and online self (train per series)
- Hyperparameter tuning and dropout for overfitting prevention
- Visualization of results (predictions, anomalies, training loss, etc.)
- Input/output via tab-separated files for easy integration
- Common utilities and dataset splitting scripts

## Technologies Used
- Python 3.x
- Keras & TensorFlow (deep learning)
- NumPy, pandas (data processing)
- Matplotlib (visualization)
- Jupyter Notebook (experimentation)
- Linux (development and execution environment)

## File Structure
- `project-algorithms.ipynb`: Main Jupyter notebook with all experiments and analysis
- `forecast.py`: LSTM-based forecasting tool
- `detect.py`: LSTM autoencoder-based anomaly detection tool
- `reduce.py`: Convolutional autoencoder for dimensionality reduction
- `split_dataset.py`: Utility to split datasets into train/test/query sets
- `common_utils.py`: Shared utility functions
- `doc/`: Documentation and assignment description

## Usage
### Input Format
Input files are tab-separated, with each row representing a time series:
```
item_id1 X11 X12 ... X1d
item_idN XN1 XN2 ... XNd
```

### Forecasting
Train and evaluate an LSTM model for stock price prediction:
```
python3 forecast.py -d <dataset path> -n <number of time series selected> -t <"offline_all" or "online_all" or "online_self">
```
- `-t offline_all`: Load pre-trained model for all series
- `-t online_all`: Train on all series during execution
- `-t online_self`: Train per series during execution

### Anomaly Detection
Detect anomalies in stock time series using LSTM autoencoders:
```
python3 detect.py -d <dataset path> -n <number of time series selected> -t <"offline_all" or "online_all"> -mae <error value as double>
```
- `-t offline_all`: Load pre-trained model for all series
- `-t online_all`: Train on all series during execution
- `-mae`: Mean Absolute Error threshold for anomaly detection

### Dimensionality Reduction & Representation Learning
Learn compact representations with convolutional autoencoders:
```
python3 reduce.py -d <dataset> -q <queryset> -od <output_dataset_file> -oq <output_query_file> -t <"offline_all" or "online_all">
```
- Outputs reduced representations in tab-separated files
- Enables clustering and similarity search

### Dataset Splitting
Split a dataset into train/test/query sets:
```
python3 split_dataset.py <dataset_path>
```

## Experimental Results
- Extensive experiments were conducted for all models, with hyperparameter tuning (batch size, layers, units, time steps, latent dimension, etc.)
- Dropout layers (p=0.2) are used to prevent overfitting
- Best models were selected based on validation loss and training curves
- See the Jupyter notebook and documentation for detailed results and analysis

## Experimental Results & Discussion

### Forecasting Time Series Values
Experiments were conducted with various hyperparameters (batch size, layers, units, time steps) for LSTM forecasting models. The optimal number of epochs was selected based on train/test loss plots to avoid overfitting. The final model used:
- time_steps = 60
- epochs = 20
- batch_size = 128
- layers = 9
- units per layer = 64

Increasing batch size reduced execution time, but very large batch sizes (>1024) did not further minimize loss. More units per layer improved error, while the number of layers and time_steps had less impact.

### Anomaly Detection in Time Series
LSTM autoencoders with dropout were used to prevent overfitting. Experiments showed:
- More layers reduced error, requiring more complex models.
- Number of units per layer had less impact on loss.
- Final model: time_steps = 60, epochs = 20, batch_size = 128, layers = 6, units per layer = 32.

### Convolutional Autoencoding of Time Series
Convolutional autoencoders were used for compression and representation. Experiments showed:
- Changing the number of layers significantly affected loss.
- Number of units and window length had less impact.
- Final model: window_length = 50, epochs = 100, batch_size = 32, 3 downsampling/3 upsampling layers, filter size = 16, latent dimension = 7.

### Conclusions
Hyperparameters were selected for optimal validation performance and to avoid overfitting. The final models balance accuracy and computational efficiency. Results and plots confirm the effectiveness of the approaches for forecasting, anomaly detection, and time series compression.

## Getting Started
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/project-algorithms-ml.git
   cd project-algorithms-ml
   ```
2. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

# Development
The project was developed in the Google Collaboratory environment as a Python notebook, which is provided. Additionally, the source code is delivered as .py files according to the project requirements. Collaboration during development was conducted in a private repository.

# Execution
The deliverable files are executed as follows:
```
python3 forecast.py -d <dataset path> -n <number of time series selected> -t <offline_all|online_all|online_self>
python3 detect.py -d <dataset path> -n <number of time series selected> -t <offline_all|online_all> -mae <error value as double>
python3 reduce.py -d <dataset> -q <queryset> -od <output_dataset_file> -oq <output_query_file> -t <offline_all|online_all>
```
About the `-t` parameter:
- `offline_all`: Load the offline pre-trained model for all time series.
- `online_all`: Train on all time series during execution.
- `online_self`: Train per time series during execution.

The directory structure must not be changed for successful execution. When running the .py files (not the .ipynb notebook), warnings may appear regarding future deprecation of certain TensorFlow functions. These can be ignored as they do not affect the current execution.

# File List
The project consists of the following files:
- `split_dataset.py`: Splits the dataset into `dataset.csv` and `queryset.csv` in the same directory.
- `common_utils.py`: Contains common utility functions for all project modules.
- `forecast.py`: Loads arguments, dataset, and depending on the `-t` parameter, either loads a pre-trained model, trains on all time series, or trains per time series. Performs time series forecasting.
- `detect.py`: Loads arguments, dataset, and depending on the `-t` parameter, either loads a pre-trained model or trains on all time series. Performs anomaly detection.
- `reduce.py`: Loads arguments, datasets, and depending on the `-t` parameter, either loads a pre-trained model or trains on all time series. Produces and visualizes compressed/encoded time series and exports them to output files.
- `project-algorithms.ipynb`: The main notebook containing all the above.

# Time Series Forecasting
All experiments were conducted for a large number of epochs. The optimal number of epochs was selected to minimize loss before overfitting occurred, as determined by the training and validation loss plots (function `plot_training_loss`).

The experiments are summarized in the following table:
| Experiment | Batch Size | Layers | Units/Layer | Time Steps | Train Loss   | Val Loss   |
|------------|------------|--------|-------------|------------|-------------|------------|
| 1          | 32         | 4      | 32          | 50         | 8.3730e-04  | 0.0807     |
| 2          | 32         | 4      | 64          | 60         | 5.3747e-04  | 0.0847     |
| 3          | 32         | 4      | 128         | 50         | 5.7512e-04  | 0.0919     |
| 4          | 64         | 6      | 64          | 60         | 5.6178e-04  | 0.0865     |
| 5          | 64         | 6      | 128         | 50         | 5.5015e-04  | 0.0909     |
| 6          | 64         | 6      | 32          | 60         | 8.6226e-04  | 0.0900     |
| 7          | 64         | 8      | 64          | 50         | 5.7873e-04  | 0.0828     |
| 8          | 128        | 8      | 128         | 60         | 4.4686e-04  | 0.0857     |
| 9          | 128        | 6      | 64          | 50         | 5.7284e-04  | 0.0855     |

Each layer consists of an LSTM layer and a dropout layer with a dropout rate of 0.2, preventing overfitting. Observations:
- Execution time decreases significantly as batch size increases, but with very large batch sizes (>1024), loss does not further decrease.
- Increasing the number of hidden layers does not reduce loss; the problem does not require a more complex model than those tested.
- The number of units per layer has a significant effect; increasing this hyperparameter reduces loss.
- The time-steps (look-back) hyperparameter does not significantly affect loss.
- The final model uses the hyperparameters from experiment 8, which are optimal.

# Time Series Anomaly Detection
As with forecasting, the optimal number of epochs was selected for anomaly detection. The experiments are summarized below:
| Experiment | Batch Size | Layers | Units | Time Steps | Train Loss | Val Loss |
|------------|------------|--------|-------|------------|------------|----------|
| 1          | 32         | 2      | 64    | 50         | 0.2111     | 0.6069   |
| 2          | 32         | 2      | 128   | 60         | 0.2120     | 0.6148   |
| 3          | 32         | 2      | 256   | 50         | 0.2110     | 0.6074   |
| 4          | 64         | 2      | 64    | 60         | 0.2143     | 0.6132   |
| 5          | 64         | 4      | 128   | 50         | 0.2112     | 0.6121   |
| 6          | 128        | 6      | 64    | 60         | 0.1987     | 0.6241   |
| 7          | 128        | 6      | 128   | 50         | 0.1998     | 0.6218   |
| 8          | 128        | 6      | 32    | 60         | 0.1975     | 0.6243   |

Each layer consists of an LSTM layer and a dropout layer with a dropout rate of 0.2, preventing overfitting.
Observations:
- Execution time decreases significantly as batch size increases, but with very large batch sizes (>1024), loss does not further decrease.
- Increasing the number of hidden layers significantly reduces loss; the problem requires a fairly complex model.
- The number of units per layer has little effect; increasing this hyperparameter does not strongly affect loss.
- The time-steps (look-back) hyperparameter does not significantly affect loss.
- The final model uses the hyperparameters from experiment 8, which are optimal.

# Convolutional Autoencoding of Time Series
As with forecasting and anomaly detection, the optimal number of epochs was selected. The experiments are summarized below:
| Experiment | Batch Size | Layers | Filter Size | Window Length | Latent Dimension | Train Loss | Val Loss |
|------------|------------|--------|-------------|--------------|------------------|------------|----------|
| 1          | 32         | 2      | 16          | 10           | 3                | 0.6791     | 0.7576   |
| 2          | 32         | 2      | 16          | 50           | 13               | 0.5563     | 0.2912   |
| 3          | 32         | 2      | 32          | 50           | 13               | 0.5303     | -0.1815  |
| 4          | 32         | 3      | 16          | 7            | 0.5317           | -0.5217    |          |
| 5          | 567        | 3      | 16          | 10           | 2                | 0.5436     | -0.9232  |
| 6          | 64         | 3      | 32          | 50           | 7                | 0.5304     | -0.8348  |
| 7          | 64         | 3      | 64          | 10           | 2                | 0.5295     | -0.2726  |
| 8          | 32         | 4      | 16          | 50           | 4                | 0.5307     | -0.1027  |
| 9          | 32         | 4      | 32          | 50           | 4                | 0.6665     | -0.8822  |

Each layer consists of a conv1d layer and a dimension upsampling/downsampling layer. For layers=3, this means 3 downsampling (conv1d and maxpooling1d) and 3 upsampling (conv1d and upsampling) layers.
Observations:
- Execution time decreases significantly as batch size increases, but with very large batch sizes (>1024), loss does not further decrease.
- Significant increase or decrease in the number of layers prevents loss minimization.
- The number of units per layer has little effect; increasing this hyperparameter does not strongly affect loss.
- The time-steps (look-back) hyperparameter does not significantly affect loss.
- The final model uses the hyperparameters from experiment 4, which are optimal.
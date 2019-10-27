import matplotlib
matplotlib.use('Agg')
import numpy as np

def compute_rmse(y, y_hat):
    differences = y - y_hat  # error.
    squared_differences = differences ** 2  # squares
    mse = np.mean(squared_differences)  # mean
    rmse_val = np.sqrt(mse)  # root
    return rmse_val


def compute_errorRate(y, y_hat):
    return np.count_nonzero(y - y_hat)/float(len(y))

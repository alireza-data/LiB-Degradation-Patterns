#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def plot_capacity_data(folder_path):
    """
    Plot capacity data from multiple .txt files in the specified folder.

    Parameters:
    - folder_path (str): Path to the folder containing .txt files.

    Returns:
    None
    """
    files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]

    plt.figure(figsize=(2 * plt.rcParams['figure.figsize'][0], 1.1 * plt.rcParams['figure.figsize'][1]))

    for file in files:
        file_path = os.path.join(folder_path, file)

        with open(file_path, 'r') as f:
            lines = f.readlines()[1:]
            x = [float(line.split()[1]) for line in lines]
            y = [float(line.split()[3]) for line in lines]

        plt.plot(x, y, label=file, linestyle='-', alpha=0.5, linewidth=2)

    plt.xlabel('Cycle Number', fontsize=18)
    plt.ylabel('Capacity/mAh', fontsize=18)
    plt.title(f'Graph of .txt files in {folder_path}')
    plt.legend(loc='lower center')

# Specify the folder path relative to the project root
folder_path = 'Data/Capacity-data'

# Call the function to plot capacity data
plot_capacity_data(folder_path)

import os
import matplotlib.pyplot as plt

def plot_voltage_data(file_path):
    """
    Plot voltage data from a specified .txt file.

    Parameters:
    - file_path (str): Path to the .txt file.

    Returns:
    None
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()[1:]
        x = [float(line.split()[0]) * 258 for line in lines]
        y = [float(line.split()[4]) for line in lines]

    plt.plot(x, y, label='Ewe/V', fillstyle='none', linewidth=1)
    plt.ylim(2.7, 4.5)
    
    plt.text(0.5, -0.2, "Voltage / Time change during charge/discharge at 25°C (25C05 cell)",
             transform=plt.gca().transAxes, ha='center', fontsize=18, weight='bold')

    plt.xlabel('Time/s')
    plt.ylabel('Ewe/V')
    plt.title('Voltage')
    plt.legend()
    plt.show()

# Specify the file name and folder path relative to the project root
file_name = "Data_Capacity_25C04.txt"
folder_path = 'Data/Capacity-data'
file_path = os.path.join(folder_path, file_name)

# Call the function to plot voltage data
plot_voltage_data(file_path)

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

def load_and_plot_gaussian_process(folder_path, filenames, title):
    """
    Load data, train a Gaussian process model, and plot the results.

    Parameters:
    - folder_path (str): Path to the folder containing data files.
    - filenames (list): List of data file names.
    - title (str): Title for the plot.

    Returns:
    None
    """
    data = {}
    for filename in filenames:
        file_path = os.path.join(folder_path, filename)
        array_data = np.loadtxt(file_path)
        data[filename] = array_data

    X_train = data['EIS_data.txt']
    Y_train = data['Capacity_data.txt']

    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
    gpr.fit(X_train, Y_train)

    EIS_data_35C02 = data['EIS_data_35C02.txt']
    X_test_35C02 = EIS_data_35C02

    Y_test_cap_35C02, Y_test_cap_35C02_var = gpr.predict(X_test_35C02, return_std=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    normalized_capacity = Y_test_cap_35C02 / Y_test_cap_35C02[0] + np.sqrt(Y_test_cap_35C02_var) / Y_test_cap_35C02[0]

    ax.fill_between(np.arange(2, 302, 2), normalized_capacity[:150], normalized_capacity[:150][::-1],
                    color=(255 / 255, 191 / 255, 200 / 255), alpha=0.5)
    ax.plot(np.arange(2, 600, 2), data['capacity35C02.txt'].flatten() / data['capacity35C02.txt'][0],
            'x', color=(0 / 255, 130 / 255, 216 / 255), linewidth=3)
    ax.plot(np.arange(2, 302, 2), Y_test_cap_35C02[:150] / Y_test_cap_35C02[0],
            '+', color=(205 / 255, 39 / 255, 70 / 255), linewidth=3)

    ax.set_xlim(0, 300)
    ax.set_ylim(0.6, 1.0)
    ax.set_xlabel('Cycle Number', fontsize=15)
    ax.set_ylabel('Identified Capacity', fontsize=15)
    ax.set_title(title, fontsize=15)
    ax.legend(['', 'Measured', 'Estimated'], loc='best', fontsize=15)
    ax.grid(True)

    plt.text(0.5, -0.2, f"Capacity estimation ({title} cell). The curves show the estimated (blue) and measured (red) capacity for the cell. R²=0.81",
             transform=plt.gca().transAxes, ha='center', fontsize=18, weight='bold')

    plt.show()

# Specify folder path and filenames relative to the project root
folder_path = 'Path_to_folder'
filenames = ['EIS_data.txt', 'Capacity_data.txt', 'EIS_data_35C02.txt', 'capacity35C02.txt']

# Call the function to load, train, and plot Gaussian process results
load_and_plot_gaussian_process(folder_path, filenames, '35C02')


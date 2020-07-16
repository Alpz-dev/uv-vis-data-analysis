import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import random


class Data(object):
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.peaks = []

    # Very basic plotting to test data changes
    def plot(self, figure, title='', start_x=None, end_x=None):
        try:
            tmp_y_data = self.y[int(start_x - self.x[0]):int((end_x + 1) - self.x[0])]
        except TypeError:
            tmp_y_data = self.y
        try:
            tmp_x_data = self.x[int(start_x - self.x[0]):int((end_x + 1) - self.x[0])]
        except TypeError:
            tmp_x_data = self.x
        plt.figure(figure)
        plt.title(str(title))
        return plt.plot(tmp_x_data, tmp_y_data)

    # Finds the average value of y across an x range
    def mean(self, start_x=None, end_x=None):
        try:
            tmp_data = self.y[int(start_x - self.x[0]):int((end_x + 1) - self.x[0])]
        except TypeError:
            tmp_data = self.y
        return np.mean(tmp_data)

    # Finds standard deviation of y across an x range
    def std(self, start_x=None, end_x=None):
        try:
            tmp_data = self.y[int(start_x - self.x[0]):int((end_x + 1) - self.x[0])]
        except TypeError:
            tmp_data = self.y
        return np.std(tmp_data)

    # Finds the signal to noise ratio of y across an x range
    # Compares the mean of the data (taken as the ground truth) to the standard deviation
    # Note: ideal for flat or constant sections of data (possible to use 1st or 2nd derivative if no constant section
    # in base data
    def snr(self, start_x=None, end_x=None):
        mean = abs(self.mean(start_x=start_x, end_x=end_x))
        std = self.std(start_x=start_x, end_x=end_x)
        SNR = mean / std
        return SNR / (1 + SNR)

    # Applies the savgol filter from the SciPy package across an x range
    def savgol_filter(self, window, order, deriv=0, start_x=None, end_x=None):
        deriv_pass = deriv
        try:
            tmp_data = self.y[int(start_x - self.x[0]):int((end_x + 1) - self.x[0])]
        except TypeError:
            tmp_data = self.y
        return Data(self.x, np.array(signal.savgol_filter(tmp_data, window, order, deriv=deriv_pass)))

    # Finds the error between two data sets across an x range
    def err(self, edited_data, start_x=None, end_x=None):
        try:
            start_i = int(start_x - self.x[0])
            end_i = int((end_x + 1) - self.x[0])
        except TypeError:
            start_i = 0
            end_i = len(self.x) - 1
        err = 0
        for i in range(start_i, end_i + 1):
            r = (self.y[i] - edited_data.y[i]) ** 2
            err += r
        return err

    # Finds the nearest odd of an integer
    # Defaults to nearest odd <= integer
    def nearest_odd(self, int, round_down=True):
        if int % 2 == 1:
            return int
        else:
            if round_down == True:
                return int - 1
            else:
                return int + 1

    # Adds simulated noise to a data set using a random gaussian distribution
    def sim_noise(self, level):
        tmp_vals = []
        for i in range(len(self.y)):
            tmp_vals.append(random.gauss(self.y[i], self.y[i] * (level / 100)))
        return Data(self.x, tmp_vals)

    # Naive search for the optimized values for the savgol filter function
    # Finds all possible values and finds the smallest measure
    # Measure = Error / (SNR^2) <- minimize error and maximize signal to noise ratio
    # Note: Can be slow for very large ranges
    def optimized_smooth(self, err_range, snr_range):
        measures = []
        index = []
        testVals = []
        for i in range(3, self.nearest_odd((len(self.x)) // 20), 2):
            for j in range(1, min(i - 1, 5)):
                tmp_smooth_data = self.savgol_filter(i, j)
                tmp_smooth_data_deriv = self.savgol_filter(i, j, deriv=1)
                err = self.err(tmp_smooth_data, start_x=err_range[0], end_x=err_range[1])
                SNR = tmp_smooth_data_deriv.snr(start_x=snr_range[0], end_x=snr_range[1])
                measures.append(err / (SNR ** 2))
                index.append((i, j))
                testVals.append((err, SNR))
        vals = index[measures.index(min(measures))]
        return self.savgol_filter(vals[0], vals[1])

    # Recursively finds the maximum y value within a given delta of an x value
    def isMaxinDelta(self, i, delta=200):
        first = 0
        last = len(self.x) - 1
        start = max(first, i - delta)
        end = min(last, i + delta)
        Max = self.y[i]
        print(Max, 'CHECKING', self.x[i], 'DELTA', (start, end))
        for j in range(start, end):
            if Max < self.y[j]:
                Max = self.y[j]
                print('RECURSE', Max, self.x[j])
                return self.isMaxinDelta(j, delta=delta)
        print(self.x[i], 'SUCCESS')
        self.peaks.append(self.x[i])
        return True

    # WIP
    # Locates peaks using a smooth 1st derivative
    def findLocalMaxima(self):
        deriv1 = self.savgol_filter(35, 2, deriv=1)
        for i in range(len(deriv1.y) - 1):
            y1 = deriv1.y[i]
            y2 = deriv1.y[i + 1]
            if (y1 > 0 and y2 < 0) and (self.isMaxinDelta(i)):
                # self.peaks.append(self.data['X'][i])
                pass


# Imports *.txt UV-Vis data from the Agilent ChemStation v10.0.1 software for Windows XP
# Creates a data obj containing x and y values from the file
def import_data(fileName):
    data = open(fileName, "r")
    x = []
    y = []
    for line in data.read().splitlines():
        hasAlpha = False
        for char in line:
            if char.isalpha():
                hasAlpha = True
                break
        if hasAlpha == False:
            for i in range(len(line)):
                if line[i].isspace():
                    x_val = float(line[:i])
                    y_val = float(line[i + 1:])
                    x.append(x_val)
                    y.append(y_val)
    return Data(x, y)

import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import random


class Data(object):
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.peaks = []

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

    def mean(self, start_x=None, end_x=None):
        try:
            tmp_data = self.y[int(start_x - self.x[0]):int((end_x + 1) - self.x[0])]
        except TypeError:
            tmp_data = self.y
        return np.mean(tmp_data)

    def std(self, start_x=None, end_x=None):
        try:
            tmp_data = self.y[int(start_x - self.x[0]):int((end_x + 1) - self.x[0])]
        except TypeError:
            tmp_data = self.y
        return np.std(tmp_data)

    def snr(self, start_x=None, end_x=None):
        mean = abs(self.mean(start_x=start_x, end_x=end_x))
        std = self.std(start_x=start_x, end_x=end_x)
        SNR = mean / std
        return SNR / (1 + SNR)

    def savgol_filter(self, window, order, deriv=0, start_x=None, end_x=None):
        deriv_pass = deriv
        try:
            tmp_data = self.y[int(start_x - self.x[0]):int((end_x + 1) - self.x[0])]
        except TypeError:
            tmp_data = self.y
        return Data(self.x, np.array(signal.savgol_filter(tmp_data, window, order, deriv=deriv_pass)))

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

    def nearest_odd(self, int, round_down=True):
        if int % 2 == 1:
            return int
        else:
            if round_down == True:
                return int - 1
            else:
                return int + 1

    def sim_noise(self, level):
        tmp_vals = []
        for i in range(len(self.y)):
            tmp_vals.append(random.gauss(self.y[i], self.y[i] * (level / 100)))
        return Data(self.x, tmp_vals)

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

    def findLocalMaxima(self):
        deriv1 = self.savgol_filter(35, 2, deriv=1)
        for i in range(len(deriv1.y) - 1):
            y1 = deriv1.y[i]
            y2 = deriv1.y[i + 1]
            if (y1 > 0 and y2 < 0) and (self.isMaxinDelta(i)):
                #self.peaks.append(self.data['X'][i])
                pass

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
                    try:
                        x_val = float(line[:i])
                    except ValueError:
                        pass
                    try:
                        y_val = float(line[i + 1:])
                    except ValueError:
                        pass
                    x.append(x_val)
                    y.append(y_val)
    return Data(x, y)



import random
import sys
import csv

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D


class Data(object):
    def __init__(self, x, y, file_name = None, instrument = ""):
        self.file_name = file_name
        self.x = np.array(x)
        self.y = np.array(y)
        self.x_wv = []
        for x in self.x:
            self.x_wv.append(10000/x)
        self.x_wv = np.array(self.x_wv)
        self.peaks = set()
        self.bands_data = []
        self.bands = []
        #if instrument == "shimadzu":
        #    print(instrument)
        #    for i in range(200, 902, 1):
        #        self.y[i - 200] += self.y[902 - 200] - self.y[898 - 200]
        #    for i in range(200, 719, 1):
        #        self.y[i - 200] += (self.y[722 - 200] - self.y[719 - 200])



    def __add__(self, other):
        if other is None:
            return self
        new_x = list(self.x)
        new_y = []
        for i in range(len(self.y)):
            new_y.append(self.y[i] + other.y[i])
        return Data(new_x, new_y)

    def __sub__(self, other):
        new_x = list(self.x)
        new_y = []
        for i in range(len(self.y)):
            new_y.append(self.y[i] - other.y[i])
        return Data(new_x, new_y)

    def __mul__(self, other):
        new_x = list(self.x)
        new_y = []
        for i in range(len(self.y)):
            new_y.append(self.y[i] * other.y[i])
        return Data(new_x, new_y)

   # def write(self, file):


    def summation(self, band_abs):
        current_total = None
        for i in range(len(band_abs)):
            current_total = band_abs[i] + current_total
        return current_total

    # Very basic plotting to test data changes
    def plot(self, figure, title='', label = '', start_x=None, end_x=None):
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
        return plt.plot(tmp_x_data, tmp_y_data, label = label)

    def plot3D(self, zVal, figure, title='', label = '', start_x=None, end_x = None):
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
        return plt.plot3D(tmp_x_data, tmp_y_data, zData, label = label)

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
        snr = mean / std
        return snr

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
    # WIP TRY AND TRANSLATE GRAPH LINEARLY WITH SOME DELTA (X ONLY)
    def translate(self, delta, start_x = None, end_x = None):
        try:
            start_i = int(start_x - self.x[0])
            end_i = int((end_x + 1) - self.x[0])
        except TypeError:
            start_i = 0
            end_i = len(self.x) - 1
        x_vals = list(self.x)
        y_vals = list(self.y)

        for i in range(start_i, end_i + 1):

            y_vals[i] += delta
        translatedData = Data(x_vals, y_vals)
        return translatedData

    def scale(self, scalar):
        x_vals = list(self.x)
        y_vals = list(self.y)
        for i in range(len(y_vals)):
            y_vals[i] *= scalar
        return Data(x_vals, y_vals)






    # Finds the nearest odd of an integer
    # Defaults to nearest odd <= integer
    def nearest_odd(self, int, round_down=True):
        if int % 2 == 1:
            return int
        else:
            if round_down:
                return int - 1
            else:
                return int + 1

    def normalize(self, start_x = None, end_x = None):
        try:
            start_i = int(start_x - self.x[0])
            end_i = int((end_x + 1) - self.x[0])
        except TypeError:
            start_i = 0
            end_i = len(self.x) - 1
        max_y_val = 0
        for i in range(start_i, end_i + 1):
            if self.y[i] >= max_y_val:
                max_y_val = self.y[i]
        y_vals = []
        x_vals = list(self.x)
        for y in self.y:
            y_vals.append(y/max_y_val)
        normalized_data = Data(x_vals, y_vals)
        return normalized_data

    def ln(self):
        y_vals = []
        x_vals = list(self.x)
        for y in self.y:
            y_vals.append(np.log(y))
        ln_data = Data(x_vals, y_vals)
        return ln_data

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
    def optimized_smooth(self, err_range, snr_range, deriv = 0):
        measures = []
        index = []
        test_vals = []
        for i in range(3, len(self.x)//40, 2):
            for j in range(2, min(i - 1, 18)):
                tmp_smooth_data = self.savgol_filter(i, j, deriv = deriv)
                tmp_smooth_data_deriv = self.savgol_filter(i, j, deriv= deriv + 1)
                err = self.err(tmp_smooth_data, start_x=err_range[0], end_x=err_range[1])
                snr = tmp_smooth_data_deriv.snr(start_x=snr_range[0], end_x=snr_range[1])
                measures.append(err-snr)
                index.append((i, j))
                test_vals.append((err, snr))
        vals = index[measures.index(min(measures))]
        print(vals)
        return self.savgol_filter(vals[0], vals[1])

    def has_delta_overlap(self, i, delta):
        for peak in self.peaks:
            if (self.x[i] < peak + delta) and (peak - delta < self.x[i]):
                return True
        return False

    # Recursively finds the maximum y value within a given delta of an x value
    def is_max_in_delta(self, i, delta=5):
        if not(self.has_delta_overlap(i, delta)):
            first = 0
            last = len(self.x) - 1
            start = max(first, i - delta)
            end = min(last, i + delta)
            maximum = self.y[i]
            for j in range(start, end):
                if maximum < self.y[j]:
                    maximum = self.y[j]
                    return self.is_max_in_delta(j, delta=delta)
            self.peaks.add(self.x[i])

    # WIP
    # Locates peaks using a smooth 1st derivative
    def find_local_maxima(self):
        deriv1 = self.savgol_filter(35, 2, deriv=1)
        for i in range(1, len(deriv1.y) - 1):
            y0 = deriv1.y[i - 1]
            y1 = deriv1.y[i]
            y2 = deriv1.y[i + 1]
            if (y1 > 0 and y2 < 0) or (y0 > 0 and y1 < 0):
                self.is_max_in_delta(i)
        return sorted(list(self.peaks))

    def derivative(self, order = 1):
        x_vals = list(self.x)
        y_vals = list(self.y)
        deriv_y = []
        for i in range(order):
            deriv_y= []
            for i in range(len(y_vals)):
                if i + 1 == len(y_vals):
                    deriv_y.append(y_vals[i])
                else:
                    new_y = (y_vals[i + 1] - y_vals[i])/1
                    deriv_y.append(new_y)
            y_vals = deriv_y
        return Data(x_vals, deriv_y)



    def find_local_maxima_v2(self, start_x = None, end_x = None):
        n_bands = []
        first_deriv_pre_smooth = self.savgol_filter(21, 2).savgol_filter(3, 1, deriv = 1)
        first_deriv_post_smooth = self.savgol_filter(3, 1, deriv = 1).savgol_filter(21, 2)
        first_deriv = first_deriv_pre_smooth + first_deriv_post_smooth
        try:
            start_i = int(start_x - self.x[0])
            end_i = int((end_x + 1) - self.x[0])
        except TypeError:
            start_i = 0
            end_i = len(self.x) - 1
        for i in range(start_i, end_i + 1):
            y0 = first_deriv.y[i - 1]
            y1 = first_deriv.y[i]
            if (y0 > 0 and y1 < 0):
                n_bands.append(i + 190)
        return n_bands

    def guassian_approximation(self, band):
        x_vals = list(self.x)
        y_vals = []
        for x in x_vals:
            a_max = band[0]
            v_max = band[1]
            v_delta = band[2]
            v = 10000 / x
            absorbance = a_max * np.exp((-((v - v_max) ** 2 / v_delta ** 2)))
            y_vals.append(absorbance)
        band_abs = Data(x_vals, y_vals)
        #band_abs.plot(1)
        return band_abs

    def nm2cm_inv(self):
        x_vals = list(self.x)
        y_vals = list(self.y)
        new_x = []
        for x in x_vals:
            cm = x/(1*10**7)
            cm_inv = 1/cm
            new_x.append(cm_inv)
        return Data(new_x, y_vals)

    def gaussian_addition(self, band_data):
        band_data_list = []
        # [A_max, v_max, v_delta]
        for band in band_data:
            band_data_list.append(self.guassian_approximation(band_data[band]))
        return self.summation(band_data_list)




    #WIP
    #Should pick out constituent peaks from the data to better identify peaks and energy transitions
    def deconvolute(self, start_x=None, end_x=None):
        n_bands = []
        first_deriv_pre_smooth = self.savgol_filter(35, 2).savgol_filter(3, 1, deriv=1)
        first_deriv_post_smooth = self.savgol_filter(3, 1, deriv=1).savgol_filter(35, 2)
        first_deriv = first_deriv_pre_smooth + first_deriv_post_smooth
        try:
            start_i = int(start_x - self.x[0])
            end_i = int((end_x + 1) - self.x[0])
        except TypeError:
            start_i = 0
            end_i = len(self.x) - 1
        for i in range(start_i, end_i + 1):
            y0 = first_deriv.y[i - 1]
            y1 = first_deriv.y[i]
            if (y0 > 0 and y1 < 0):
                n_bands.append(i)
        band_data = dict()
        for band in n_bands:
            #[A_max, v_max, v_delta]
            band_data[band] = [float(self.y[band]), float(10000/self.x[band]), 1.3]
        all_vals = []
        for band in band_data:
            all_vals.append(band_data[band][0])
            all_vals.append(band_data[band][1])
            all_vals.append(band_data[band][2])





        def deconvolute_opt(all_vals):
            global current_cost
            i = 0
            for band in band_data:
                band_data[band][0] = all_vals[i]
                band_data[band][1] = all_vals[i + 1]
                band_data[band][2] = all_vals[i + 2]
                i += 3
            summation = self.gaussian_addition(band_data)
            result = self.err(summation, start_x=start_x, end_x=end_x)
            current_cost = result
            return result

        def log_cost(all_vals):
            cost_values.append(current_cost)
            if len(cost_values) > 2:
                current_tr_radius = abs((current_cost - cost_values[-2])/2)
                tr_radius_values.append(current_tr_radius)
                if abs(current_tr_radius) > 10**(-8):
                    progress = np.arctan(1 / abs(10**(-8) - current_tr_radius))/(np.pi/2)
                    i = int(np.round(progress * 100))
                    sys.stdout.write('\r')
                    sys.stdout.write("Minimization Progress: [%-99s] %d%%" % ('=' * i, i))
                    sys.stdout.flush()




        cost_values = []
        tr_radius_values = []
        result = minimize(deconvolute_opt, np.array(all_vals),
                          method='SLSQP', callback=log_cost)
        print(result)
        all_vals = list(result.x)
        i = 0
        for band in band_data:
            band_data[band][0] = all_vals[i]
            band_data[band][1] = all_vals[i + 1]
            band_data[band][2] = all_vals[i + 2]
            i += 3
            self.bands_data.append(band_data[band])
        summation = self.gaussian_addition(band_data)
        summation.plot(1)
        for band in self.bands_data:
            data = self.guassian_approximation(band)
            self.bands.append(data)
            data.plot(1)




# Imports *.txt UV-Vis data from the Agilent ChemStation v10.0.1 software for Windows XP
# Creates a data obj containing x and y values from the file
def import_data(file_name, instr = ""):
    data = open(file_name, "r")
    x = []
    y = []
    for line in data.read().splitlines():
        has_alpha = False
        for char in line:
            if char.isalpha():
                has_alpha = True
                break
        if not has_alpha:
            for i in range(len(line)):
                if line[i].isspace():
                    x_val = float(line[:i].replace(',', ''))
                    y_val = float(line[i + 1:].replace(',', ''))
                    x.append(x_val)
                    y.append(y_val)
    return Data(x, y, file_name = file_name, instrument = instr)

def import_csv(file_name):
    file = open(file_name, 'r')
    data = csv.reader((line.replace('\0', '') for line in file), delimiter = ",")
    x = []
    y = []
    for line in data:
        if line != []:
            if line[0].isnumeric():
                x.append(float(line[0]))
                y.append(float(line[1]))
    return Data(x, y)

data = import_csv("test data/10-17-2020_UV-VIS_NI-PET_SYNTH-09-01-2020+09-02-2020_IN DCM_PTLC_03.CSV")
data2 = import_csv("test data/10-17-2020_UV-VIS_NI-PET_SYNTH-09-01-2020+09-02-2020_IN DCM_PTLC_02.CSV")
data3 = import_csv("test data/10-17-2020_UV-VIS_NI-PET_SYNTH-09-01-2020+09-02-2020_IN DCM_PTLC_01.CSV")

data.plot(1, start_x = 300, end_x = 800)
data2.plot(1, start_x = 300, end_x = 800)
data3.plot(1, start_x = 300, end_x = 800)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Abs")
plt.legend(["Ni6", "Ni5", "Ni4"])

plt.show()
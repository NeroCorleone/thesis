import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelmin

import csv
import os

directory = '/home/nefta/thesis/results/wg3_2/supercurrent/'

dirpath, dirnames, filenames = list(os.walk(directory))

def flip_minima(abs_values, real_values):
    main_peak = np.argmax(abs_values)
    left_min = [0]
    right_min = [main_peak]
    for localmin in argrelmin(abs_values, order=2):
        if localmin < main_peak:
            left_min.append(localmin)
        else:
            right_min.append(localmin)
    left_min.append(main_peak)
    right_min.appen(len(abs_values))
    left_min_rev = list(reversed(left_min))

    osc = []
    n = len(left_min)
    for i in range(n - 1):
        for index in range(left_min_rev[n  - i - 1], left_min_rev[n - i - 2]):
            osc.append((-1)**(n - i) * abs(real_values[index]))
    for i in range(n - 1):
        for index in range(right_min[i], right_min[i+1]):
            osc.append((-1)**i * abs(real_values[index]))
    return(osc)

def read_values(filename):
    result = []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            result.append([float(element) for element in row])
    return(result)

def transform_to_density(b_values, Ic_values, y_values):
    W = 360 
    L = 100 
    omega = 2 * np.pi / W
    
    phi_values = b_values * W * L 
    dPhi = phi_values[1] - phi_values[0]
    maxPhi = phi_values[-1]
    
    real, imag = Ic_values
    
    Jy_values = []
    for y in y_values:
        Jy_real = 0
        Jy_imag = 0
        for k, phi in enumerate(phi_values):
            Ic_real = real[k] #Fourier coefficient
            Ic_imag = imag[k]
            Jy_real += np.cos(np.pi * phi / (2 * maxPhi))**0.5 *(Ic_real * np.cos(omega * phi * y) - Ic_imag * np.sin(omega * phi * y)) * dPhi
            Jy_imag += np.cos(np.pi * phi / (2 * maxPhi))**0.5 *(Ic_imag * np.cos(omega * phi * y) + Ic_real * np.sin(omega * phi * y)) * dPhi
            
        Jy_abs = np.sqrt(Jy_real**2 + Jy_imag**2)
        Jy_values.append(Jy_abs)
    return(Jy_values)

for dirname in dirnames:
    filename = dirname + 'data.csv'
    try:
        values = read_values(filename)
    except:
        print('Could not open {}'.format(filename))
        continue
    abspart, realpart, imagpart = values
    realpart_flipped = flip_minima(realpart)

    fourier = transform_to_density(b_values, Ic_values, y_values
    

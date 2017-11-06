import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import numpy as np
from scipy.signal import argrelmin

import csv
import os

#directory = '/users/tkm/kanilmaz/thesis/results/wg3_1_double/supercurrent/'
directory = '/home/nefta/thesis/results/wg3_2_double/supercurrent/'

W = 360
L = 100
#directory = '/users/tkm/kanilmaz/thesis/results/hb/supercurrent/use/'
#W = 918 
#L = 204


dirpath, dirnames, filenames = list(os.walk(directory))[0]
phi_sg_values = [float(filename[4:-18]) for filename in list(os.walk(directory))[0][1]]

def flip_minima(abs_values, real_values):
    main_peak = np.argmax(abs_values) # index of main peak
    left_min = [0]
    right_min = [main_peak]
    for loc_min in argrelmin(abs_values, order=4)[0]:
        if loc_min < main_peak:
            left_min.append(loc_min)
        else: right_min.append(loc_min)
    left_min.append(main_peak)
    right_min.append(len(abs_values))
    left_min_rev = list(reversed(left_min))
    
    #print(left_min, right_min, left_min_rev)
    osc = []
    n = len(left_min)
    for i in range(n - 1):
        #print(n-i - 1, n-i-2)
        for index in range(left_min_rev[n  - i - 1], left_min_rev[n - i - 2]):
            #osc.append((-1)**(i+1) * abs(real_values[index]))
            osc.append((-1)**(n - i) * abs(real_values[index]))
    for i in range(n - 1):
        #print(i, i+1)
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

def transform_to_density(b_values, real, imag, y_values):
    omega = 2 * np.pi / W
    
    phi_values = b_values * W * L 
    dPhi = phi_values[1] - phi_values[0]
    maxPhi = phi_values[-1]
    Jy_values = []
    for y in y_values:
        Jy_real = 0
        Jy_imag = 0
        for k, phi in list(enumerate(phi_values)):
            Ic_real = real[k] #Fourier coefficient
            Ic_imag = imag[k]
            Jy_real += abs(np.cos(np.pi * phi / (2 * maxPhi)))**0.5 * (Ic_real * np.cos(omega * phi * y) - Ic_imag * np.sin(omega * phi * y)) * dPhi
            Jy_imag += abs(np.cos(np.pi * phi / (2 * maxPhi)))**0.5 * (Ic_imag * np.cos(omega * phi * y) + Ic_real * np.sin(omega * phi * y)) * dPhi
        Jy_abs = np.sqrt(Jy_real**2 + Jy_imag**2)
        Jy_values.append(Jy_abs)
    return(Jy_values)

def plot_current_and_density(phi_values, abs_values, real_flip, 
                             y_values, j_values, phi_sg, filedir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
    fig.suptitle(r'$\varphi_{{SG}} = {}$'.format(phi_sg), fontsize=20)
    ax1.plot(phi_values, abs_values, marker='o', label='abs',)
    ax1.plot(phi_values, real_flip, marker='o', label='real, flipped', )
    ax1.set_xlabel(r'$\phi / \phi_0 $', fontsize=18)
    ax1.set_ylabel(r'$I_c$', fontsize=18)
    ax1.set_xlim([phi_values[0], phi_values[-1]])
    ax1.set_ylim([min(real_flip), max(abs_values)])
    ax1.legend()
    #ax1.grid()
    
    ax2.plot(y_values, j_values, marker='o', linestyle='none')
    ax2.set_xlabel(r'$y\ [a]$', fontsize=18)
    ax2.set_ylabel(r'$J(y)$', fontsize=18)
    ax2.set_xlim(y_values[0], y_values[-1])
    #ax2.grid() #usage of ax.grid() in combination with fivethirtyeight style leads to plot without grids
    
    fig.savefig(filedir + '/current_and_density_{}.png'.format(phi_sg))
    fig.savefig(filedir + '/current_and_density_{}.ps'.format(phi_sg))
    return()

for phi_sg, dirname in list(zip(phi_sg_values, dirnames)):
    print('running..., phi = {}'.format(phi_sg))
    filename = directory + dirname + '/data.csv'
    filedirectory = directory + dirname
    paramsfile = directory + dirname + '/params.txt'
    params = {}

    with open(paramsfile) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            params[row[0]] = row[1]

    try:
        values = read_values(filename)
    except:
        print('Could not open {}'.format(filename))
        continue
    abspart, realpart, imagpart = np.asarray(values).T
    realpart_flipped = flip_minima(abspart, realpart)
    
    nbPoints = float(params['nb_points'])
    maxB = float(params['maxB']) 
    mag_field = np.linspace(-maxB, maxB, nbPoints)
    
    phi_norm = mag_field * W * L
    y = np.linspace(-1.5*W, 1.5*W, 500)
    density = transform_to_density(mag_field, realpart_flipped, imagpart, y)
    plot_current_and_density(phi_norm, abspart, realpart_flipped, 
                             y, density, phi_sg, filedirectory)
    
    

    

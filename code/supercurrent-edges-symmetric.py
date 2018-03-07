import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import kwant
import numpy as np
import scipy
import scipy.ndimage
import scipy.linalg as la
from types import SimpleNamespace
from datetime import datetime
import multiprocessing as mp
import queue
import os
from functools import partial
import csv

vsg_values = np.arange(-0.01, -0.1, -0.01) 
vbg = 0.25 
vlead = 0.0
nb_points = 300 
max_b = 0.000035
magnetic_field = np.linspace(- max_b, max_b, nb_points)
maxPhi = np.pi
phase = (-np.pi, np.pi) 

delta = 1.0 
T = delta / 20
eta = 2.5 
gamma = 0.4
at = 5.0
a = 0.4

pot_decay = 0 
mainpath = '/users/tkm/kanilmaz/thesis/' 
#mainpath = '/home/nefta/thesis/'

channelwidth = 15 
scattering_region = np.ones((210, 700))

"""
#QPC-like gate for edge transmission
path_to_result = mainpath + 'results/full_gate_edges/symmetric/supercurrent/' 
topgate = np.zeros(np.shape(scattering_region))
gatewidth = topgate.shape[1] - 2 * channelwidth
gatelength = 50
gate_pos = (round(topgate.shape[0] / 2) - round(gatelength / 2), 
            round(topgate.shape[0] / 2) + round(gatelength / 2))
for row in topgate[gate_pos[0]:gate_pos[1]]:
    row[channelwidth:-channelwidth] = np.ones(gatewidth)
"""

#Full gate covering scattering region, with edge channels
path_to_result = mainpath + 'results/edges/symmetric/supercurrent/' 
topgate = np.zeros(np.shape(scattering_region))
gatewidth = topgate.shape[1] - 2 * channelwidth
for index, row in enumerate(topgate):
    topgate[index][channelwidth:-channelwidth] = np.ones(gatewidth)

topgate_gauss = scipy.ndimage.gaussian_filter(topgate, pot_decay)

potential = scipy.interpolate.RectBivariateSpline(
    x=(a*np.arange(topgate_gauss.shape[0])),
    y=(a*np.arange(topgate_gauss.shape[1])),
    z=topgate_gauss, 
    kx=1,
    ky=1
)

bilayer =  kwant.lattice.general([(at*np.sqrt(3)/2, at*1/2), (0, at*1)],
                                 [(0, 0.0), (at*1 / (2*np.sqrt(3)), at*1/2), 
                                  (-at*1/(2*np.sqrt(3)), at*1/2), (0, 0)])

a1, b1, a2, b2 = bilayer.sublattices
hoppings1 = (((0, 0), a1, b1), ((0, 1), a1, b1), ((1, 0), a1, b1)) 
hoppings2 = (((0, 0), a2, b2), ((0, -1), a2, b2), ((1, -1), a2, b2))

def onsite(site, par):    
    potentialTop = par.v_sg * potential(site.pos[0], site.pos[1])
    mu = (par.v_bg + potentialTop) / 2 #+ Vb
    delta = - (potentialTop - par.v_bg) / eta
    # site.family in (a1, b1)
    if (site.family == a1 or site.family == b1):
        return - mu - delta #+ disorder #- edge_gap
    return -mu + delta #+ disorder  #+ edge_gap


def onsite_lead(site, par):     
    potentialTop = par.v_lead 
    mu = (par.v_bg + potentialTop) / 2
    delta = - ( potentialTop - par.v_bg) / eta 
    # site.family in (a1, b1)
    if site.family == a1 or site.family == b1:
        return - mu - delta
    return -mu  + delta

def geomShape(pos):
    #x, y = pos
    if pos[0] < 0 or pos[1] < 0:
        return False
    try:
        # rather round()?
        return scattering_region[int(pos[0] / a), int(pos[1] / a)]
    except IndexError:
        return False

def hop_intra_layer(site1, site2, par): 
    xt, yt = site1.pos 
    xs, ys = site2.pos
    return -par.t * np.exp(-0.5j * np.pi * par.B  * (xt - xs) * (yt + ys))

def hop_inter_layer(site1, site2, par): 
    return -par.gamma1 

def hop_intra_layer_lead(site1, site2, par): 
    return -par.t 

def hop_inter_layer_lead(site1, site2, par): 
    return -par.gamma1 

def leadShape1(pos):
    y = pos[1]
    if y < 0:
        return False
    try:
        return scattering_region[0, int(y / a)]
    except IndexError:
        return False
    
def leadShape2(pos):
    y = pos[1]
    if y < 0:
        return False
    try:
        return scattering_region[-1, int(y / a)]
    except IndexError:
        return False

def trs(m):
    return m.conj()

class TRIInfiniteSystem(kwant.builder.InfiniteSystem):
    def __init__(self, lead, trs):
        """A lead with time reversal invariant modes."""
        self.__dict__ = lead.__dict__
        self.trs = trs

    def modes(self, energy=0, args=()):
        prop_modes, stab_modes =             super(TRIInfiniteSystem, self).modes(energy=energy, args=args)
        n = stab_modes.nmodes
        stab_modes.vecs[:, n:(2*n)] = self.trs(stab_modes.vecs[:, :n])
        stab_modes.vecslmbdainv[:, n:(2*n)] =             self.trs(stab_modes.vecslmbdainv[:, :n])
        prop_modes.wave_functions[:, n:] =             self.trs(prop_modes.wave_functions[:, :n])
        return prop_modes, stab_modes

def make_system():
    system = kwant.Builder()
    scat_width= scattering_region.shape[0]
    scat_length = scattering_region.shape[1]
    
    system[bilayer.shape(geomShape, (0.5*a*scat_width, 0.5*a*scat_length))] = onsite 
    system[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings1]] = hop_intra_layer
    system[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings2]] = hop_intra_layer
    system[kwant.builder.HoppingKind((0, 0), a1, b2) ] = hop_inter_layer    
      
    trans_sym_1 = kwant.TranslationalSymmetry(bilayer.vec((-2, 1)))
    first_lead = kwant.Builder(trans_sym_1)
    first_lead[bilayer.shape(leadShape1, (0, 0.5*a*scat_length))] = onsite_lead
    first_lead[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings1]] = hop_intra_layer_lead
    first_lead[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings2]] = hop_intra_layer_lead
    first_lead[kwant.builder.HoppingKind((0, 0), a1, b2)] = hop_inter_layer_lead
    
    trans_sym_2 = kwant.TranslationalSymmetry(bilayer.vec((2, -1))) #?
    second_lead = kwant.Builder(trans_sym_2)
    second_lead[bilayer.shape(leadShape2, (0, 0.5*a*scat_length))] = onsite_lead
    second_lead[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings1]] = hop_intra_layer_lead
    second_lead[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings2]] = hop_intra_layer_lead
    second_lead[kwant.builder.HoppingKind((0, 0), a1, b2)] = hop_inter_layer_lead
     
    system.attach_lead(first_lead)
    system.attach_lead(second_lead)
    system = system.finalized()
    system.leads = [TRIInfiniteSystem(lead, trs) for lead in system.leads]
    
    return system

def superCurrent(scatMatrix, phi):
    nbModes = [len(leadInfo.momenta) for leadInfo in scatMatrix.lead_info]
    scatData = scatMatrix.data
    
    dim1 = int(nbModes[0]/2)
    dim2 = int(nbModes[1]/2)
    
    r_a11 = 1j*np.eye(dim1)
    r_a12 = np.zeros((dim1, dim2))
    r_a21 = r_a12.T
    r_a22 = 1j*np.exp(- 1j * phi) * np.eye(dim2)
    r_a = np.bmat([[r_a11, r_a12], [r_a21, r_a22]])
    
    A = (r_a.dot(scatData) + (scatData.T).dot(r_a)) / 2
    
    dr_a11 = np.zeros((dim1, dim1))
    dr_a12 = np.zeros((dim1, dim2))
    dr_a21 = dr_a12.T
    dr_a22 = np.exp(- 1j * phi) * np.eye(dim2)
    dr_a = np.bmat([[dr_a11, dr_a12], [dr_a21, dr_a22]])

    dA = (dr_a.dot(scatData) + (scatData.T).dot(dr_a)) / 2
    
    dA_total = np.array((dA.T.conj()).dot(A) + (A.T.conj()).dot(dA))
    
    eigenval, eigenvec = la.eigh(A.T.conj().dot(A))
    
    #finalEigenVal = delta * eigenVal ** 0.5 
    finalEigenVal = delta * np.sqrt(eigenval)
    finalEigenVec = eigenvec.T
    
    current_complex =  np.sum(
        (vec.T.conj().dot(dA_total.dot(vec)) * np.tanh(val/T)/val)
        for val, vec in zip(finalEigenVal, finalEigenVec)
    )
    #current = 0.5 * delta ** 2 * np.real(current_complex)
    imag = 0.5 * delta ** 2 * np.imag(current_complex)
    real = 0.5 * delta ** 2 * np.real(current_complex)
    absval = 0.5 * delta ** 2 * np.abs(current_complex)
    
    return (absval, real, imag)

def findMax(func, phase_min, phase_max):
    current = [func(phi) for phi in np.linspace(phase_min, phase_max)]
    currentPeak = max(current, key=lambda x: x[0])
    #currentPeak = np.amax(current)
    return currentPeak


def maxCurrent(system, params):
    (pos, par) = params
    scatMatrix = kwant.smatrix(system, energy=0.0, args=[par])
    func = partial(superCurrent, scatMatrix )
    currentPeak = findMax(func, phase[0], phase[-1])
    return((pos, currentPeak))

def worker(system, param_queue, result_queue):
    try:
        while True:
            params = param_queue.get(block=False)
            result = maxCurrent(system, params)
            result_queue.put(result)
            param_queue.task_done()
    except queue.Empty:
        pass

def plotCurrentPerV(magneticField, current, filename):
    plt.figure(figsize=(10, 8))
    plt.plot(current, magneticField, linestyle='None', marker='o', color='b', )
    plt.xlabel(r'$B$', fontsize=14)
    plt.ylabel(r'$I_c$', fontsize=14)
    plt.savefig(filename)
    return


def current_vs_b(system, vsg, path=path_to_result):
    runtime = datetime.strftime(datetime.today(), '%Y%m%d-%H:%M:%S')
    system_params_names = ['vsg', 'vbg', 'vlead', 'maxB', 'nb_points',
                           'decay', 'eta', 'gamma', 'a', 
                           'at', 'delta', 'T', ]
    system_params = [str(vsg), str(vbg), str(vlead), str(max_b), str(nb_points), 
                    str(pot_decay), str(eta), str(gamma), str(a), 
                    str(at), str(delta), str(T), ]
    
    newpath = path + 'vsg=' + str(vsg) + '-' +  runtime + '/'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    system_params_file = newpath + 'params.txt'
    with open(system_params_file, 'w' ) as paramfile:
        for name, value in zip(system_params_names, system_params):
            paramfile.write(name + ", " + value + '\n')

    runtime = datetime.strftime(datetime.today(), '%Y%m%d-%H:%M:%S')

    param_queue = mp.JoinableQueue()
    result_queue = mp.JoinableQueue() 
    param_args = []
    
    for i, b in enumerate(magnetic_field):
        param_args.append((i, SimpleNamespace(v_bg=vbg, v_lead=vlead, t=1, gamma1=gamma, v_sg=vsg, B=b)))
    for arg in param_args:
        param_queue.put(arg)
        
    timestamp = datetime.now()
    print('starting calculation with ', len(magnetic_field),' points')
    
    nb_cores = mp.cpu_count()
    processes = [mp.Process(target=worker, args=(system, param_queue, result_queue)) for i in range(nb_cores)]
    
    for p in processes:
        p.start()
    
    param_queue.join()
    
    results = []
    try:
        while True:
            results.append(result_queue.get(block=False))
    except queue.Empty:
        pass
    print('time for calculation with multiprocessing: ', datetime.now() - timestamp)    
    sorted_results = sorted(results, key=lambda value: value[0])
    unzipped = list(zip(*sorted_results))
    current_values = np.asarray(unzipped[1])
        
    filename = newpath + 'data.csv' 
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')
        for row in current_values:
            writer.writerow(list(row))
    pngfile = newpath + 'v_sg=' + str(vsg) + '.png'
    plotCurrentPerV(current_values.T[0], magnetic_field, pngfile)
    print('output in', filename)
    return()

system = make_system()
def family_colors(site):
    delta = 1.0 - potential(site.pos[0], site.pos[1])[0][0]
    if delta < 0.85:    
        return('black')
    else:
        return('white')
    
#fig = kwant.plotter.plot(system, fig_size=(16, 9), site_color=family_colors)
#fig.savefig('test.png')

for vsg in vsg_values:
    current_vs_b(system, vsg)


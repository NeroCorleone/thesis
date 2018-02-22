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

vsg_values = [0.0, ]#np.arange(-0.0, -0.1, -0.01)
vbg = 0.2 
vdis = 0 
vlead = 0.0
nb_points = 500 
max_b = 0.00005
magnetic_field = np.linspace(- max_b, max_b, nb_points)
maxPhi = np.pi
phase = (-np.pi, np.pi) 

delta = 1.0 
T = delta / 20
eta = 2.5 
gamma = 0.4
at = 5
a = 0.4

pot_decay = 15 
mainpath = '/users/tkm/kanilmaz/thesis/'
#mainpath = '/home/nefta/thesis/'

path_to_result = mainpath + 'results/qpc/supercurrent/rough/' 
path_to_file = mainpath +'designfiles/qpc.png'
path_to_scatfile = mainpath +'designfiles/scatteringRegion.png'
topgate = 1 - scipy.ndimage.imread(path_to_file, mode='L').T / 255
topgate_gauss = scipy.ndimage.gaussian_filter(topgate, pot_decay)
scattering_region = np.fliplr(1 - scipy.ndimage.imread(
                            path_to_scatfile, mode='L').T / 255) 
#depth = 20
#size = 0.5

# ### armchair edge

bilayer =  kwant.lattice.general([(at*np.sqrt(3)/2, at*1/2), (0, at*1)],
                                 [(0, 0.0), (at*1 / (2*np.sqrt(3)), at*1/2), 
                                  (-at*1/(2*np.sqrt(3)), at*1/2), (0, 0)])

a1, b1, a2, b2 = bilayer.sublattices
hoppings1 = (((0, 0), a1, b1), ((0, 1), a1, b1), ((1, 0), a1, b1)) 
hoppings2 = (((0, 0), a2, b2), ((0, -1), a2, b2), ((1, -1), a2, b2))

potential = scipy.interpolate.RectBivariateSpline(
    x=(a*np.arange(topgate_gauss.shape[0])),
    y=(a*np.arange(topgate_gauss.shape[1])),
    z=topgate_gauss,
    kx=1,
    ky=1,
)

def onsite(site, par):    
    topgate_potential = par.v_sg * potential(site.pos[0], site.pos[1]) 
    mu = (par.v_bg + topgate_potential) / 2 
    delta = - (topgate_potential - par.v_bg) / eta 
    disorder = par.v_dis * np.random.uniform(0, 1)
    if (site.family == a1 or site.family == b1):
        return - mu - delta + disorder
    return -mu + delta + disorder

def onsite_lead(site, par):     
    topgate_potential = 0
    mu = (par.v_bg + topgate_potential) / 2
    delta = - ( topgate_potential - par.v_bg) / eta
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
        prop_modes, stab_modes = super(TRIInfiniteSystem, self).modes(energy=energy, args=args)
        n = stab_modes.nmodes
        stab_modes.vecs[:, n:(2*n)] = self.trs(stab_modes.vecs[:, :n])
        stab_modes.vecslmbdainv[:, n:(2*n)] = self.trs(stab_modes.vecslmbdainv[:, :n])
        prop_modes.wave_functions[:, n:] = self.trs(prop_modes.wave_functions[:, :n])
        return prop_modes, stab_modes


def make_edges_rough(system, depth, size, lead_distance=2):
    site_positions = [site.pos for site in system.sites()]
    unique_x = np.unique(list(zip(*site_positions))[0])[lead_distance:-lead_distance]
    ymin = {xval: min([val for val in site_positions if val[0] == xval], key=lambda x: x[1])[1] for xval in unique_x}
    ymax = {xval: max([val for val in site_positions if val[0] == xval], key=lambda x: x[1])[1] for xval in unique_x}
    
    def upper_edge(site, width):
        x0, y0 = site.pos
        try:
            delta = ymax[x0] - y0
            if delta < width:
                return(True)
        except KeyError:
            return False

    def lower_edge(site, width=10):
        x0, y0 = site.pos
        try:
            delta = y0 - ymin[x0]
            if delta < width:
                return(True)
        except KeyError:
            return False
    
    upper_edge_sites = [(site.tag, site.family) for site in system.sites() if upper_edge(site, depth)]
    upper_indices_to_delete = np.random.choice(np.arange(len(upper_edge_sites)),
                                               round(len(upper_edge_sites) * size))
    
    for index in upper_indices_to_delete:
        tag_to_del, family_to_del = upper_edge_sites[index]
        try:
            del system[family_to_del(tag_to_del[0], tag_to_del[1])]
        except KeyError:
            pass
    
    lower_edge_sites = [(site.tag, site.family) for site in system.sites() if lower_edge(site, depth)]
    lower_indices_to_delete = np.random.choice(np.arange(len(lower_edge_sites)), 
                                               round(len(lower_edge_sites)*size))
    
    for index in lower_indices_to_delete:
        tag_to_del, family_to_del = lower_edge_sites[index]
        try:
            del system[family_to_del(tag_to_del[0], tag_to_del[1])]
        except KeyError:
            pass
    return(system)


def make_system(depth, size):
    system = kwant.Builder()
    scat_width = scattering_region.shape[0]
    scat_length = scattering_region.shape[1]

    system[bilayer.shape(geomShape, (0.5*a*scat_width, 0.5*a*scat_length))] = onsite 
    system[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings1]] = hop_intra_layer
    system[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings2]] = hop_intra_layer
    system[kwant.builder.HoppingKind((0, 0), a1, b2) ] = hop_inter_layer    

    trans_sym_1 = kwant.TranslationalSymmetry(bilayer.vec((-2, 1)))
    lead_1 = kwant.Builder(trans_sym_1)
    lead_1[bilayer.shape(leadShape1, (0, 0.5*a*scat_length))] = onsite_lead
    lead_1[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings1]] = hop_intra_layer_lead
    lead_1[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings2]] = hop_intra_layer_lead
    lead_1[kwant.builder.HoppingKind((0, 0), a1, b2)] = hop_inter_layer_lead
    
    trans_sym_2 = kwant.TranslationalSymmetry(bilayer.vec((2, -1))) #?
    lead_2 = kwant.Builder(trans_sym_2)
    lead_2[bilayer.shape(leadShape2, (0, 0.5*a*scat_length))] = onsite_lead
    lead_2[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings1]] = hop_intra_layer_lead
    lead_2[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings2]] = hop_intra_layer_lead
    lead_2[kwant.builder.HoppingKind((0, 0), a1, b2)] = hop_inter_layer_lead
    
    system.attach_lead(lead_1)
    system.attach_lead(lead_2)
    system = make_edges_rough(system, depth, size)
    system = system.finalized()
    system.leads = [TRIInfiniteSystem(lead, trs) for lead in system.leads]#
    return(system)

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
    #return absval

def findMax(func, phase_min, phase_max):
    current = [func(phi) for phi in np.linspace(phase_min, phase_max)]
    currentPeak = max(current, key=lambda x: x[0])
    #currentPeak = np.amax(current)
    return(currentPeak)


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
    system_params_names = ['vsg', 'vbg', 'vdis', 'vlead', 'maxB', 'nb_points',
                           'decay', 'eta', 'gamma', 'a', 
                           'at', 'delta', 'T', ]
    system_params = [str(vsg), str(vbg), str(vdis), str(vlead), str(max_b), str(nb_points), 
                    str(pot_decay), str(eta), str(gamma), str(a), 
                    str(at), str(delta), str(T), ]
    
    newpath = path + 'depth=' + str(depth) + 'size=' + str(size) +  '-' +  runtime + '/'
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
        param_args.append((i, SimpleNamespace(v_bg=vbg, v_lead=vlead, t=1, gamma1=gamma, v_sg=vsg, B=b, v_dis=vdis)))
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

for depth, size in [(5, 0.5), (15, 0.5), (25, 0.5), (35, 0.5),]: 
    system = make_system(depth, size)
    for vsg in vsg_values:
        current_vs_b(system, vsg)


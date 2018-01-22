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

vbg_values = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4] 
nb_points = 400 
vsg_values = np.linspace(-1, 0.0, nb_points)
vlead = 0.0
b = 0.0

delta = 1.0 
T = delta / 20
eta = 2.5 
gamma = 0.4
at = 5.0
a = 0.2

pot_decay = 15 
mainpath = '/users/tkm/kanilmaz/thesis/'

path_to_result = mainpath + 'results/zigzagedge/qpc/conductance/' 
path_to_file = mainpath +'designfiles/qpc.png'
path_to_scatfile = mainpath +'designfiles/scatteringRegion.png'
topgate = 1 - scipy.ndimage.imread(path_to_file, mode='L').T / 255
scattering_region = np.fliplr(1 - scipy.ndimage.imread(
                            path_to_scatfile, mode='L').T / 255) 

#path_to_result = mainpath + 'results/wg3_2/supercurrent/' 
#path_to_file = mainpath +'designfiles/waveguide3_2_small.png'
#topgate = 1 - scipy.ndimage.imread(path_to_file, mode='L').T / 255
#scattering_region = np.ones(topgate.shape)
"""
case = 'wg3_2'
setups = {'wg3_2': ('results/wg3_2/supercurrent/', 'designfiles/waveguide3_2_small.png')}

path_to_result, path_to_file = (mainpath + setups[case][0], mainpath + setups[case][1]) 

read_files = {
        'wg3_2': scipy.ndimage.imread(mainpath + setups['wg3_2'][1], mode='L') / 255, 
        }

topgate = 1 - read_files[case]

scat_file = mainpath + 'designfiles/scatteringRegions.png'

scattering_cases = {
        'wg3_2': np.ones(topgate.shape)
        }

scattering_region = scattering_cases[case]
"""
topgate_gauss = scipy.ndimage.gaussian_filter(topgate, pot_decay)

potential = scipy.interpolate.RectBivariateSpline(
    x=(a*np.arange(topgate_gauss.shape[0])),
    y=(a*np.arange(topgate_gauss.shape[1])),
    z=topgate_gauss, 
    kx=1,
    ky=1
)

#bilayer with zigzag edges
sin30, cos20 = sin30, cos30 = (1/2, np.sqrt(3)/2)
zigzag = kwant.lattice.general([(at*1, 0), (at*sin30, at*cos30)],
                                 [(0, 0), (0, at/np.sqrt(3)),
                                  (0, 0), (at/2, at/(2*np.sqrt(3)))])

a1, b1, a2, b2 = zigzag.sublattices
#different hoppings for zigzag edges
hoppings1 = (((0, 0), a1, b1), ((-1, 1), a1, b1), ((0, 1), a1, b1))
hoppings2 = (((0, 0), a2, b2), ((1, 0), a2, b2), ((0, 1), a2, b2))

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
    scat_width, scat_length = scattering_region.shape

    sys = kwant.Builder()
    sys[zigzag.shape(geomShape, (0.5*a*scat_width, 0.5*a*scat_length))] = onsite
    sys[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings1]] = hop_intra_layer
    sys[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings2]] = hop_intra_layer
    sys[kwant.builder.HoppingKind((0, 0), a1, b2) ] = hop_inter_layer

    sym1 = kwant.TranslationalSymmetry(zigzag.vec((-1,0)))
    sym1.add_site_family(a1, other_vectors=[(-1, 2)])
    sym1.add_site_family(b1, other_vectors=[(-1, 2)])
    sym1.add_site_family(a2, other_vectors=[(-1, 2)])
    sym1.add_site_family(b2, other_vectors=[(-1, 2)])

    lead_1 = kwant.Builder(sym1)
    lead_1[zigzag.shape(leadShape1, (0, 0.5*a*scat_length))] = onsite_lead
    lead_1[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings1]] = hop_intra_layer_lead
    lead_1[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings2]] = hop_intra_layer_lead
    lead_1[kwant.builder.HoppingKind((0, 0), a1, b2)] = hop_inter_layer_lead

    sym2 = kwant.TranslationalSymmetry(zigzag.vec((1, 0)))
    sym2.add_site_family(a1, other_vectors=[(1, -2)])
    sym2.add_site_family(b1, other_vectors=[(1, -2)])
    sym2.add_site_family(a2, other_vectors=[(1, -2)])
    sym2.add_site_family(b2, other_vectors=[(1, -2)])

    lead_2 = kwant.Builder(sym2)
    lead_2[zigzag.shape(leadShape2, (0, 0.5*a*scat_length))] = onsite_lead
    lead_2[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings1]] = hop_intra_layer_lead
    lead_2[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings2]] = hop_intra_layer_lead
    lead_2[kwant.builder.HoppingKind((0,0), a1, b2)] = hop_inter_layer_lead

    sys.attach_lead(lead_1)
    sys.attach_lead(lead_2)
    sys = sys.finalized()
    sys.leads = [TRIInfiniteSystem(lead, trs) for lead in sys.leads]
    return sys 

def conductance(system, params):
    pos, param = params
    smatrix = kwant.smatrix(system, energy=0.0, args=[param])
    conductance = smatrix.transmission(1, 0)
    return (pos, conductance)

def worker(system, param_queue, result_queue):
    try:
        while True:
            params = param_queue.get(block=False)
            result = conductance(system, params)
            result_queue.put(result)
            param_queue.task_done()
    except queue.Empty:
        pass

def plotConductance(splitgate, conductance, filename):
    plt.figure(figsize=(16, 9))
    plt.plot(splitgate, conductance, marker='o', color='b', )
    plt.xlabel(r'$\varphi_{SG}$', fontsize=18)
    plt.ylabel(r'Conductance $[2e/ \hbar]$', fontsize=18)
    plt.savefig(filename)
    return


def calculate_conductance(system, vbg, path=path_to_result):
    runtime = datetime.strftime(datetime.today(), '%Y%m%d-%H:%M:%S')
    system_params_names = ['vbg', 'vlead', 'b', 'nb_points',
                           'decay', 'eta', 'gamma', 'a', 
                           'at', 'delta', 'T', ]
    system_params = [str(vbg), str(vlead), str(b), str(nb_points), 
                    str(pot_decay), str(eta), str(gamma), str(a), 
                    str(at), str(delta), str(T), ]
    
    newpath = path + 'vbg=' + str(vbg) + '-' +  runtime + '/'
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
    
    for i, vsg in enumerate(vsg_values):
        param_args.append((i, SimpleNamespace(v_bg=vbg, v_lead=vlead, t=1, gamma1=gamma, v_sg=vsg, B=b)))
    for arg in param_args:
        param_queue.put(arg)
        
    timestamp = datetime.now()
    print('starting calculation with ', len(vsg_values),' points')
    
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
    print('time for calculation: ', datetime.now() - timestamp)    
    sorted_results = sorted(results, key=lambda value: value[0])
    unzipped = list(zip(*sorted_results))
    conductance_values = np.asarray(unzipped[1])
     
    filename = newpath + 'data.csv' 
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')
        writer.writerow(conductance_values)
        #for row in current_values:
        #    writer.writerow(list(row))
    pngfile = newpath + 'v_bg=' + str(vbg) + '.png'
    plotConductance(vsg_values, conductance_values, pngfile)
    print('output in', filename)
    return()

system = make_system()

for vbg in vbg_values:
    calculate_conductance(system, vbg)


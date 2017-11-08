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

nb_points = 500 
splitgate_voltage = np.linspace(0.0, 1.0, nb_points)
vbg_values = [-0.2, -0.3, -0.4, -0.5, ]#[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]#np.round(np.linspace(0.4, 0.9, 5), 2) 
phase = (-np.pi, np.pi) 

delta = 1.0 
T = delta / 2
eta = 2.5 
gamma = 0.4
at = 5 
a = 0.4

pot_decay = 0 
case = 'wg3_2'
mainpath = '/users/tkm/kanilmaz/thesis/'
setups = {'hb': ('results/hb/conductance/', 'designfiles/halfBarrier.png'),
          'hb_lower': ('results/hb_lower/conductance/', 'designfiles/hb_lower_part.png'),
          'qpc': ('results/qpc/conductance/', 'designfiles/qpc_gate.png'), 
          #'wg3_1': ('results/wg3_1/conductance/', 'designfiles/waveguide3_1.png')
          'wg3_1': ('results/wg3_1/conductance/', 'designfiles/waveguide3_1_small.png'),
          'wg3_2': ('results/wg3_2/conductance/', 'designfiles/waveguide3_2_small.png'),
          'wg3_1_double': ('results/wg3_1_double/conductance/', 'designfiles/waveguide3_double_1_small.png'),
          'wg3_2_double': ('results/wg3_2_double/conductance/', 'designfiles/waveguide3_double_2_small.png'),
          }

path_to_result, path_to_file = (mainpath + setups[case][0], mainpath + setups[case][1])

read_files = {'hb': scipy.ndimage.imread(mainpath + setups['hb'][1], mode='L').T / 255,
        'hb_lower': scipy.ndimage.imread(mainpath + setups['hb'][1], mode='L').T / 255,
        'qpc': scipy.ndimage.imread(mainpath + setups['qpc'][1])[:,:,0].T / 255,
        'wg3_1': scipy.ndimage.imread(mainpath + setups['wg3_1'][1], mode='L') / 255,
        'wg3_2': scipy.ndimage.imread(mainpath + setups['wg3_2'][1], mode='L') / 255,
        'wg3_1_double': scipy.ndimage.imread(mainpath + setups['wg3_1_double'][1], mode='L') / 255,
        'wg3_2_double': scipy.ndimage.imread(mainpath + setups['wg3_2_double'][1], mode='L') / 255,
        }

topgate = 1 - read_files[case]

#topgate = 1 - scipy.ndimage.imread(
#        '/users/tkm/kanilmaz/code/half_barrier/designfiles/hb_lower_part.png', mode='L').T / 255

scat_file = mainpath + 'designfiles/scatteringRegion.png'

scattering_cases = {'hb': 1 - scipy.misc.imread(scat_file)[:,:,0].T / 255,
                    'hb_lower': 1 - scipy.misc.imread(scat_file)[:,:,0].T / 255,
                    'qpc': 1 - scipy.misc.imread(scat_file)[:, :, 0].T / 255,
                    'wg3_1': np.ones(topgate.shape),
                    'wg3_2': np.ones(topgate.shape),
                    'wg3_1_double': np.ones(topgate.shape),
                    'wg3_2_double': np.ones(topgate.shape),
        }
#scattering_region = 1 - scipy.misc.imread(mainpath + 'designfiles/scatteringRegion.png')[:, :, 0].T / 255
scattering_region = scattering_cases[case]

topgateGauss = scipy.ndimage.gaussian_filter(topgate, pot_decay)

potential = scipy.interpolate.RectBivariateSpline(
    x=(a*np.arange(topgateGauss.shape[0])),
    y=(a*np.arange(topgateGauss.shape[1])),
    z=topgateGauss, 
    kx=1,
    ky=1,
)

bilayer =  kwant.lattice.general([(at*np.sqrt(3)/2, at*1/2), (0, at*1)],
                                 [(0, 0.0), (at*1 / (2*np.sqrt(3)), at*1/2), 
                                  (-at*1/(2*np.sqrt(3)), at*1/2), (0, 0)])
a1, b1, a2, b2 = bilayer.sublattices
hoppings1 = (((0, 0), a1, b1), ((0, 1), a1, b1), ((1, 0), a1, b1)) 
hoppings2 = (((0, 0), a2, b2), ((0, -1), a2, b2), ((1, -1), a2, b2))

def onsite(site, par):    
    topgate_potential = par.v_sg * potential(site.pos[0], site.pos[1]) 
    mu = (par.v_bg + topgate_potential) / 2 
    delta = - (topgate_potential - par.v_bg) / eta 
    # site.family in (a1, b1)
    if (site.family == a1 or site.family == b1):
        return - mu - delta 
    return -mu + delta

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
        prop_modes, stab_modes =             super(TRIInfiniteSystem, self).modes(energy=energy, args=args)
        n = stab_modes.nmodes
        stab_modes.vecs[:, n:(2*n)] = self.trs(stab_modes.vecs[:, :n])
        stab_modes.vecslmbdainv[:, n:(2*n)] =             self.trs(stab_modes.vecslmbdainv[:, :n])
        prop_modes.wave_functions[:, n:] =             self.trs(prop_modes.wave_functions[:, :n])
        return prop_modes, stab_modes

def make_system():
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
    system = system.finalized()
    system.leads = [TRIInfiniteSystem(lead, trs) for lead in system.leads]
    
    return(system)


def conductance(system, params):
    pos, param = params
    smatrix = kwant.smatrix(system, energy=0.0, args=[param])
    conductance = smatrix.transmission(1, 0)
    #conductance = smatrix.transmission(0, 0)
    return((pos, conductance))

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
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(splitgate, conductance, marker='o', color='b', )
    ax.set_xlabel(r'$V_{SG}$', fontsize=18)
    ax.set_ylabel('Conductance', fontsize=18)
    #ax.set_yscale('log')
    fig.savefig(filename)
    return

def calculate_conductance(system, vbg, b=0, path=path_to_result):
    runtime = datetime.strftime(datetime.today(), '%Y%m%d-%H:%M:%S')
    system_params_names = ['vbg', 'b', 'nb_points', 
			   'decay', 'eta', 'gamma', 
                           'a', 'at', 'delta', 'T', ]
    system_params = [str(vbg), str(b), str(nb_points), 
		     str(pot_decay), str(eta), str(gamma), 
                     str(a), str(at), str(delta), str(T), ]
    newpath = path + 'vbg=' + str(vbg) + '-' +  runtime + '/'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    system_params_file = newpath + 'params.txt'
    with open(system_params_file, 'w' ) as paramfile:
        for name, value in zip(system_params_names, system_params):
            paramfile.write(name + ", " + value + '\n')

    param_queue = mp.JoinableQueue()
    result_queue = mp.JoinableQueue() 
    param_args = []

    for i, vsg in enumerate(splitgate_voltage):
        param_args.append((i, SimpleNamespace(v_sg=vsg, v_bg=vbg, t=1, gamma1=gamma, B=b)))
    for arg in param_args:
        param_queue.put(arg)

    timestamp = datetime.now()
    print('starting calculation with ', str(nb_points), 'points')
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
        
    png_file = newpath + 'conductance.png'
    plotConductance(splitgate_voltage, conductance_values, png_file)
    return()

sys = make_system()
fig_sys = kwant.plotter.plot(sys)
fig_sys.savefig('{}-system.png'.format(case))
for vbg in vbg_values:
    print('vbg=', vbg)
    calculate_conductance(sys, vbg)
'''
for vsg in [-0.2, -0.3, -0.4, -0.45, -0.46, -0.47, -0.48, -0.49, -0.5, ]: 
    print(vsg)
    par = SimpleNamespace(v_bg=v_bg, t=1, gamma1=gamma, v_sg=vsg, B=0.0)
    ldos = kwant.ldos(sys, energy=0.0, args=[par])
    fig = kwant.plotter.map(system, ldos, a=1, vmax=.1, vmin=0.02, fig_size=(10,20))
    dir = mainpath + '/{}/ldos/'.format(case) 
    if not os.path.exists(dir):
        os.makedirs(dir)
    fig.savefig(dir + 'ldos_vsg={0}.png'.format(vsg))

'''

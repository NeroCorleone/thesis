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
import scipy.sparse.linalg as sla

nb_points = 100 
max_b = 0.00005
magnetic_field = np.linspace(- max_b, max_b, nb_points)
vsg_values = [-0.8, ]#[-0.2, -0.3, -0.325, -0.35, -0.375, -0.4, -0.45, -0.5, -0.55, -0.6]
backgate_voltage = 0.25 

### needed for current calculation and make_system()
maxPhi = np.pi
phase = (-np.pi, np.pi) 

delta = 1.0 
T = delta / 2
eta = 2.5 
gamma = 0.4

at = 5.0
a = 0.4

topgate = 1 - scipy.ndimage.imread(
        '/users/tkm/kanilmaz/thesis/designfiles/qpc.png')[:, :, 0].T / 255
    #'/users/tkm/kanilmaz/thesis/designfiles/waveguide3_2_small.png', mode='L').T / 255
    #'/users/tkm/kanilmaz/Dropbox/master/delft_code/supercurrent/BLG-SNS/code/QPC_device_top_gate.png')[:, :, 0].T / 255
    #'/home/nefta/Dropbox/master/delft_code/supercurrent/BLG-SNS/code/QPC_device_top_gate.png')[:, :, 0].T / 255

scatteringGeom = 1 - scipy.misc.imread(
        '/users/tkm/kanilmaz/thesis/designfiles/scatteringRegion.png')[:,:,0].T / 255
#        '/users/tkm/kanilmaz/Dropbox/master/code/halfBarrier/designfiles/scatteringRegion.png')[:, :, 0].T / 255
    #'/home/nefta/Dropbox/master/code/halfBarrier/designfiles/scatteringRegion.png')[:, :, 0].T / 255
#scatteringGeom = np.ones(topgate.shape)
topgateGauss = scipy.ndimage.gaussian_filter(topgate, 20)

potential = scipy.interpolate.RectBivariateSpline(
    x=(a*np.arange(topgateGauss.shape[0])),
    y=(a*np.arange(topgateGauss.shape[1])),
    z=topgateGauss, 
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
    potentialTop = 0
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
        return scatteringGeom[int(pos[0] / a), int(pos[1] / a)]
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
        return scatteringGeom[0, int(y / a)]
    except IndexError:
        return False
    
def leadShape2(pos):
    y = pos[1]
    if y < 0:
        return False
    try:
        return scatteringGeom[-1, int(y / a)]
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
    
    #width, length = scatt....shape
    scatWidth = scatteringGeom.shape[0]
    scatLength = scatteringGeom.shape[1]
    
    #potential for scattering region
    system[bilayer.shape(geomShape, (0.5*a*scatWidth, 0.5*a*scatLength))] = onsite 
    #hopping matrix elements for hopping within layer (intra) or hopping between two layers (inter) (?)
    system[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings1]] = hop_intra_layer
    system[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings2]] = hop_intra_layer
    system[kwant.builder.HoppingKind((0, 0), a1, b2) ] = hop_inter_layer    
      
    #define first lead
    transSym1 = kwant.TranslationalSymmetry(bilayer.vec((-2, 1)))
    firstLead = kwant.Builder(transSym1)
    firstLead[bilayer.shape(leadShape1, (0, 0.5*a*scatLength))] = onsite_lead
    firstLead[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings1]] = hop_intra_layer_lead
    firstLead[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings2]] = hop_intra_layer_lead
    firstLead[kwant.builder.HoppingKind((0, 0), a1, b2)] = hop_inter_layer_lead
    
    #define second lead
    transSym2 = kwant.TranslationalSymmetry(bilayer.vec((2, -1))) #?
    secondLead = kwant.Builder(transSym2)
    secondLead[bilayer.shape(leadShape2, (0, 0.5*a*scatLength))] = onsite_lead
    secondLead[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings1]] = hop_intra_layer_lead
    secondLead[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings2]] = hop_intra_layer_lead
    secondLead[kwant.builder.HoppingKind((0, 0), a1, b2)] = hop_inter_layer_lead
     
    system.attach_lead(firstLead)
    system.attach_lead(secondLead)
    system = system.finalized()
    system.leads = [TRIInfiniteSystem(lead, trs) for lead in system.leads]
    
    return system

### Supercurrent calculation
def superCurrent(scatMatrix, delta, T, phi):
    nbModes = [len(leadInfo.momenta) for leadInfo in scatMatrix.lead_info]
    #print(nbModes)
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
    
    eigenVal, eigenVec = la.eigh(A.T.conj().dot(A))
    
    finalEigenVal = delta * eigenVal ** 0.5 
    finalEigenVec = eigenVec.T
    
    currentImag =  np.sum(
        (vec.T.conj().dot(dA_total.dot(vec)) * np.tanh(val/T)/val)
        for val, vec in zip(finalEigenVal, finalEigenVec)
    )
    current = 0.5 * delta ** 2 * np.real(currentImag)
    
    return current

def findMax(func, phase_min, phase_max):
    current = [func(phi) for phi in np.linspace(phase_min, phase_max)]
    currentPeak = np.amax(current)
    return(currentPeak)


def maxCurrent(system, params):
    (phase, delta, T, namespace_arg) = params
    pos, par = namespace_arg
    scatMatrix = kwant.smatrix(system, energy=0.0, args=[par])
    func = partial(superCurrent, scatMatrix, delta, T)
    currentPeak = findMax(func, phase[0], phase[-1])
    return((pos,currentPeak))

def worker(system, param_queue, result_queue):
    try:
        while True:
            params = param_queue.get(block=False)
            result = maxCurrent(system, params)
            result_queue.put(result)
            param_queue.task_done()
    except queue.Empty:
        pass

def eigenvalues(system, params):
    count, par = params
    hamilton = system.hamiltonian_submatrix(args=[params], sparse=True)
    ev = sla.eigsh(hamilton, k=15, which='SM', return_eigenvectors=False)
    return(ev)

def worker_bands(system, param_queue, result_queue):
    try:
        while True:
            params = param_queue.get(block=False)
            result = eigenvalues(system, params)
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

#### Creating system
system = make_system()

def current_vs_b(vsg):
    ### Create directory to store data in:
    runtime = datetime.strftime(datetime.today(), '%Y%m%d-%H:%M:%S')
    path = '/users/tkm/kanilmaz/code/qpc/current_vs_b/' 
    #path = '/home/nefta/code/qpc/current_vs_b/' 
    system_params_names = ['vsg', 'backgate_voltage', 'maxB', 'nb_points', 'eta', 'gamma', 'a', 'at', 'delta', 'T', ]
    system_params = [str(vsg), str(backgate_voltage), str(max_b), str(nb_points), str(eta), str(gamma), str(a), str(at), str(delta), str(T), ]
    
    #newpath = path + system_params + runtime + '/'
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
    namespace_args = []
    count = 0
    
    for b in magnetic_field:
        namespace_args.append((count, SimpleNamespace(v_bg=backgate_voltage, t=1, gamma1=gamma, v_sg=vsg, B=b)))
        count +=1
    for arg in namespace_args:
        param_queue.put((phase, delta, T, arg))
        
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
    print(current_values)
        
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')
        writer.writerow(current_values)
                
    print('output in', filename)
        
    pngfile = newpath + 'v_sg=' + str(vsg) + '.png'
    plotCurrentPerV(current_values, magnetic_field, pngfile)
    return()

for vsg in vsg_values:
    current_vs_b(vsg)


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import kwant
import scipy.ndimage
import numpy as np
from types import SimpleNamespace
import os

dirname = '/home/nefta/thesis/results/bands/'
topgate = 1- scipy.ndimage.imread(
        '/home/nefta/thesis/designfiles/waveguide3_2_small.png', mode='L') / 255

#dirname = '/users/tkm/kanilmaz/thesis/results/bands/'
#topgate = 1- scipy.ndimage.imread(
#        '/users/tkm/kanilmaz/thesis/designfiles/waveguide3_2_small.png', mode='L') / 255

scatteringGeom = np.ones(topgate.shape)
vbg = 0.5
vlead = 0.3 
vsg_values = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7]#np.arange(0.0, -1, -0.1) 
b = 0.0
gamma = 0.4
a = 0.4
at = 5.0
bilayer = kwant.lattice.general([(at*np.sqrt(3)/2, at*1/2), (0, at*1)], [(0, 0.0), (at*1 / (2*np.sqrt(3)), at*1/2), (-at*1/(2*np.sqrt(3)), at*1/2), (0, 0)])


a1, b1, a2, b2 = bilayer.sublattices

hoppings1 = (((0, 0), a1, b1), ((0, 1), a1, b1), ((1, 0), a1, b1)) 
hoppings2 = (((0, 0), a2, b2), ((0, -1), a2, b2), ((1, -1), a2, b2))

'''
def onsite_lead(site, par):   
    mu = (par.v_bg + par.v_sg) / 2
    delta = - ( par.v_sg - par.v_bg) / 2.5
    # site.family in (a1, b1)
    if (site.family == a1 or site.family == b1):
        return - mu - delta
    return -mu  + delta
'''
def onsite_lead(site, par):   
    topgate_potential = par.v_sg + par.v_lead 
    mu = (par.v_bg + topgate_potential) / 2
    delta = - ( topgate_potential - par.v_bg) / 2.5
    # site.family in (a1, b1)
    if (site.family == a1 or site.family == b1):
        return - mu - delta
    return -mu  + delta

def hop_intra_layer(site1, site2, par): 
    xt, yt = site1.pos 
    xs, ys = site2.pos
    return -par.t * np.exp(-0.5j * np.pi * par.B  * (xt - xs) * (yt + ys))

def hop_inter_layer(site1, site2, par): 
    return -par.gamma1


def leadShape1(pos):
    y = pos[1]
    if y < 0:
        return False
    try:
        return scatteringGeom[0, int(y/a)]
    except IndexError:
        return False

def make_system_lead():
    scatLength = scatteringGeom.shape[1]
    transSym = kwant.TranslationalSymmetry(bilayer.vec((-2, 1)))
    lead = kwant.Builder(transSym)
    lead[bilayer.shape(leadShape1, (0, 0.5*a*scatLength))] = onsite_lead
    lead[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings1]] = hop_intra_layer
    lead[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings2]] = hop_intra_layer
    lead[kwant.builder.HoppingKind((0, 0), a1, b2)] = hop_inter_layer
    return(lead) 


def calculate_bands(par):
    system_params_names = ['vsg', 'vbg', 'b', 'gamma', 'a', 'at', ]
    system_params = [str(par.v_sg), str(par.v_bg), str(par.B),  str(par.gamma1), str(a), str(at), ]
    resultdir = dirname + 'vsg={0}-vbg={1}-vlead={2}/'.format(par.v_sg, par.v_bg, par.v_lead)
    if not os.path.exists(resultdir):
        os.makedirs(resultdir)
    system_params_file = resultdir + 'params.txt'
    with open(system_params_file, 'w' ) as paramfile:
        for name, value in zip(system_params_names, system_params):
            paramfile.write(name + ", " + value + '\n')

    bands = kwant.physics.Bands(sys_lead.finalized(), args=[par])
    momenta = np.linspace(-np.pi, np.pi, 50)
    energies = [bands(k) for k in momenta]
    band_index = int(len(energies[0]) / 2)

    start = band_index - 3
    stop = band_index + 3
    fig2, ax = plt.subplots(figsize=(8, 6))
    for i in range(start, stop):
        band = [energy[i] for energy in energies]
        ax.plot(momenta, band, label=str(i))
        ax.set_xlabel("Momentum k", fontsize=14)
        ax.set_ylabel("Energies",  fontsize=14)

    ax.set_title(r'Band structure for $\varphi_{{BG}} = {0}$, $\varphi_{{SG}} = {1}$, $\varphi_{{LD}} ={2}$'.format(par.v_bg, par.v_sg, par.v_lead), fontsize=16)
    ax.grid()
    #ax.legend(loc=0)
    fig2.savefig(resultdir + 'bands_vbg={0}_vsg={1}_b={2}.png'.format(par.v_bg, par.v_sg, par.B))
    return()

sys_lead = make_system_lead()

for vsg in vsg_values:
    print(vsg)
    par = SimpleNamespace(v_bg=vbg, v_sg=vsg, v_lead=vlead, t=1, gamma1=gamma, B=b)
    calculate_bands(par)

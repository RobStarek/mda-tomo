# -*- coding: utf-8 -*-
"""
Small collection of functions for plotting single-qubit states on Bloch
sphere in Hammer projection.
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

def KetToBloch(ket):
    if ket[0,0] == 0:
        return (np.pi,0)
    elif ket[1,0] == 0:
        return (0, 0)        

    phi = np.angle(ket[1,0])-np.angle(ket[0,0])
    phi = np.angle(np.exp(1j*phi))
    theta = np.arccos(np.abs(ket[0,0]))*2
    return theta, phi

def RhoToBloch(rho):
    """
    Naive approach, good for pure states
    """
    if rho[0,0] == 0:
        return (np.pi,0)
    elif rho[1,1] == 0:
        return (0, 0) 

    phi = np.angle(rho[1,0])    
    theta = np.arccos(rho[0,0]**0.5)*2
    return theta, phi



def RhoToBloch2(rho):
    """
    Projection on pure states
    """
    if rho[0,0] == 0:
        return (np.pi,0)
    elif rho[1,1] == 0:
        return (0, 0) 

    eigs, vects = np.linalg.eigh(rho)
    p = np.max(eigs) - np.min(eigs) #mixing factor
    isel = np.argmax(eigs)
    
    if p==0:
        raise("Maximally mixed state, unable to project onto sphere.")

    arg = ((rho[0,0] - 0.5 + 0.5*p)/p)**0.5
    phi = np.angle(rho[1,0])
    theta = np.arccos(arg)*2
    return theta, phi

def BlochToGeo(theta, phi):
    return (np.pi/2 - theta), phi

def KetToHammer(ket):
    return Hammer(*BlochToGeo(*KetToBloch(ket)))

def RhoToHammer(rho):
    return Hammer(*BlochToGeo(*RhoToBloch(rho)))

def RhoToHammer2(rho):
    return Hammer(*BlochToGeo(*RhoToBloch2(rho)))

def Hammer(lattitude, longitude, wrapback=False):
    clon = np.cos(longitude/2)
    slon = np.sin(longitude/2)
    clat = np.cos(lattitude)
    slat = np.sin(lattitude)
    sqrt2 = 2**.5
    denom = (1+clat*clon)**.5

    x = 2*sqrt2*clat*slon/denom
    #wrap small deviations back    
    if ((x+2*sqrt2*clat/denom) < 0.01) and wrapback:
        delta = (x+2*sqrt2*clat/denom)
        x = -x + delta

    y = sqrt2*slat/denom

    return x.real, y.real

def PlotHammerGrid(ax, Nx, Ny):
    #Generate grid
    lats = np.linspace(-np.pi/2, np.pi/2, Ny)
    lons = np.linspace(-np.pi, np.pi, Nx)
    gridxy = np.zeros((Ny, Nx, 2))
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            gridxy[i,j] = Hammer(lat, lon)

    #equators
    for i in range(Ny):
        ax.plot(gridxy[i,:,0], gridxy[i,:,1],"-", color='black', lw=0.5)
    #meridians
    for j in range(Nx):
        ax.plot(gridxy[:,j,0], gridxy[:,j,1],"-", color='black', lw=0.5)

def PlotHammerKets(ax, kets, fmt = 'o', clr='red', label=None):
    coords = np.array([KetToHammer(ket) for ket in kets])
    ax.plot(coords[:,0],coords[:,1],fmt, color=clr, label=label)

def PlotHammerKetsAnno(ax, kets, fmt = 'o', clr='red', label=None):
    coords = np.array([KetToHammer(ket) for ket in kets])

    ax.plot(coords[:,0],coords[:,1],fmt, color=clr, label=label)
    for i, (x, y) in enumerate(coords):
        ax.text(x, y, f'{i:02d}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color=clr)

def PlotHammerRhos(ax, rhos, fmt = 'o', clr='red', label=None):
    coords = np.array([RhoToHammer2(rho) for rho in rhos])
    ax.plot(coords[:,0],coords[:,1],fmt, color=clr, label=label)


# import KetSugar as ks
# states = [ks.LO, ks.HI, ks.HLO, ks.HHI, ks.CLO, ks.CHI]
# rhos = [ks.ketbra(ket, ket) for ket in states]
# fig, ax = plt.subplots(1,1)
# PlotHammerGrid(ax, 11, 11)
# PlotHammerKets(ax, states, 'o','red')
# PlotHammerRhos(ax, rhos, '+','blue')
# plt.show()

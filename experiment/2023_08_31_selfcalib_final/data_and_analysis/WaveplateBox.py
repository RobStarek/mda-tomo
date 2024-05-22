# -*- coding: utf-8 -*-
"""
Toolbox of commonly used function for waveplate-polarization manipulation.
Jones formalism and Bloch-sphere representation is used.
"""


import numpy as np
from scipy.optimize import minimize

def ROT(x):
    """
    Matrix of 2D coordinate rotation with angle x.    
    """
    cx = np.cos(x)
    sx = np.sin(x)
    return np.array([[cx, -sx],[sx, cx]])

def HWP(x):
    """
    Rotated half-wave plate in Jones formalism.
    """
    M = np.array([[1,0],[0,-1]])
    R = ROT(x)
    return R @ M @ R.T

def QWP(x):
    """
    Rotated quarter-wave plate in Jones formalism.
    """    
    M = np.array([[1,0],[0,-1j]])
    R = ROT(x)
    return R @ M @ R.T

def WP(x, gamma):
    """
    Rotated gamma-wave plate in Jones formalism.
    """    
    M = np.array([[1,0],[0,np.exp(-1j*gamma)]])
    R = ROT(x)
    return R @ M @ R.T


def WPPrepare(x,y, input_state = np.array([[1],[0]]), dret1=0, dret2=0):
    """
    Jones vector (ket) of polarization prepared from input_state through a 
    pair of wave plates.
    input_state-->QWP-->HWP-->output
    Args:
        x - rotation of half-wave plate
        y - rotation of quarter-wave plate
        input_state - Jones (ket) vector of input state
        dret1 - retardance error of half-wave plate
        dret2 - retardance error of quarter-wave plate
    Returns:
        Jones vector of the prepared state
    """
    if dret1!=0 or dret2!=0:
        return WP(x, np.pi + dret1) @ WP(y, np.pi/2 + dret2) @ input_state
    else:
        return HWP(x) @ QWP(y) @ input_state


def WPProj(x,y, proj_state = np.array([[1],[0]]), dret1=0, dret2=0):   
    """
    Waveplate projector state. ->HWP(x)->QWP(y)->POL
    Args:
        x ... rotation angle of half-wave plate (rad)
        y ... rotation angle of quarter-wave plate
        proj_state ... used polarization filter, its ket representation
        dret1 ... retardance of HWP
        dret2 ... retardance of QWP
    Returns:
        <bra| vector of projecting state
    """ 
    return (proj_state.T.conjugate()) @ WP(y, np.pi/2 + dret2) @ WP(x, np.pi + dret1)


def QHQBlock(x,y,z, dgx=0, dgy=0, dgz=0):
    """
    Unitary of QWP-HWP-QWP block.
    Order: --QWP(z)->HWP(y)->QWP(x)->
    Args:
        x - rotation of quarter-wave plate
        y - rotation of half-wave plate
        z - rotation of quarter-wave plate
        dgx, dgy, dgz - respective error retardances        
    Returns:
        unitary matrix of net effect
    """    
    #return QWP(x) @ HWP(y) @ QWP(z)
    return WP(x, dgx+np.pi/2) @ WP(y, dgy+np.pi) @ WP(z, dgz+np.pi/2)

def ProcessSimilarity(A,B):
    """
    Measure similarity of two trace-preserving single-qubit operations.
    Choi-Jamiolkovski isomorphism (channel-state duality)
    is used to represent unitary operations as vectors.

    Args:
        A, B - 2x2 ndarrays of unitary operation
    Returns:
        Process fidelity - 0 to 1, with 0 for orthogonal processes, 1 for identical processes.
    """
    Bell = np.array([[1],[0],[0],[1]])*2**-.5
    BellA = np.kron(np.eye(2), A) @ Bell
    BellB = np.kron(np.eye(2), B) @ Bell
    return np.abs(BellA.T.conjugate() @ BellB)[0,0]**2

"""Auxilliary QWP-HWP-QWP and HWP-QWP Search-grids for minimization
This is required for SearchForU and SearchForKet functions"""
deg = np.pi/180
grid_qwp = np.array([0.1, 89.1, -89.9])*deg
grid_hwp = np.array([0.1, 44.9, -45.1, 89.1, -89.1])*deg
search_grid_qhq = []
search_grid_qh = []
wp_bounds = (-185*deg, 185*deg)
for alpha in grid_qwp:
    for beta in grid_hwp:
        search_grid_qh.append([beta, alpha])
        for gamma in grid_qwp:
            search_grid_qhq.append([alpha, beta, gamma])            

def SearchForU(U, tol=1e-6):
    """
    Search for angles x,y,z which implement desired unitary U with
    three wave-plates:
    QWP(X)-HWP(y)-QWP(z)->    
    In this version, only perfect waveplates are considered.
    Args:
        U - desired unitary single-qubit operation
        tol - minimizer tolerance, see scipy.optimize.minimize docs.
    Returns:
        R - dictionary with minimization details. 
        R['x'] contains desired angles, R['fun'] is measure of quality (should be -1)
        See scipy.optimize.minimize docs for more details.
    """
    #Construct function to be minimized
    def minim(x):
        Ux = QHQBlock(x[0],x[1],x[2])        
        return -ProcessSimilarity(Ux, U)
    Rs = [minimize(minim, g, bounds=[wp_bounds]*3, tol=tol) for g in search_grid_qhq]
    Fs = np.array([R['fun'] for R in Rs])
    idx = np.argmin(Fs)    
    return Rs[idx]

def SearchForU2(U, dgx=0, dgy=0, dgz=0, tol=1e-6):
    """
    Search for angles x,y,z which implement desired unitary U with
    three wave-plates with imperfect retardances:
    QWP(X)-HWP(y)-QWP(z)->    
    In this version, only perfect waveplates are considered.
    Args:
        U - desired unitary single-qubit operation
        dgx, dgy, dgz - respective waveplate retardance errors
        tol - minimizer tolerance, see scipy.optimize.minimize docs.
    Returns:
        R - dictionary with minimization details. 
        R['x'] contains desired angles, R['fun'] is measure of quality (should be -1)
        See scipy.optimize.minimize docs for more details.
    """
    #Construct function to be minimized
    def minim(x):
        Ux = QHQBlock(x[0],x[1],x[2], dgx, dgy, dgz)        
        return -ProcessSimilarity(Ux, U)
    Rs = [minimize(minim, g, bounds=[wp_bounds]*3, tol=tol) for g in search_grid_qhq]
    Fs = np.array([R['fun'] for R in Rs])
    idx = np.argmin(Fs)    
    return Rs[idx] 

def SearchForKet(ket, input_state=np.array([[1],[0]]), dret1 = 0, dret2 = 2, tol=1e-6):
    """
    Search for wave plates angles x,y which prepare desired ket state from input state
    with setup:
    input_state->QWP(y)-HWP(x)->
    Args:
        ket - desired Jones vector to be prepared
        input_state - input Jones vector
        dret1 - retardance error of half-wave plate
        dret2 - retardance error of quarter-wave plate
        tol - minimizer tolerance, see scipy.optimize.minimize docs.
    Returns:
        R - dictionary with minimization details. 
        R['x'] contains desired angles, R['fun'] is measure of quality (should be -1)
        See scipy.optimize.minimize docs for more details.
    """    
    #Construct function to be minimized
    def minim(x):
        ketX = WPPrepare(x[0],x[1], input_state, dret1, dret2)
        return -np.abs(ketX.T.conjugate() @ ket)[0,0]**2
    #Start minimization from multiple initial guessses to avoid sub-optimal local extremes.
    Rs = [minimize(minim, g, bounds=[wp_bounds]*2, tol=tol) for g in search_grid_qh]
    #Pick global extereme.
    Fs = np.array([R['fun'] for R in Rs])
    idx = np.argmin(Fs)
    return Rs[idx]

def SearchForRho(rho, input_state=np.array([[1],[0]]), dret1 = 0, dret2 = 0, tol=1e-6):
    """
    Search for wave plates angles x,y which prepare desired pure density matrix from input state
    with setup:
    input_state->QWP(y)-HWP(x)->
    Args:
        ket - desired Jones vector to be prepared
        input_state - input Jones vector
        dret1 - retardance error of half-wave plate
        dret2 - retardance error of quarter-wave plate
        tol - minimizer tolerance, see scipy.optimize.minimize docs.
    Returns:
        R - dictionary with minimization details. 
        R['x'] contains desired angles, R['fun'] is measure of quality (should be -1)
        See scipy.optimize.minimize docs for more details.
    """    
    #Construct function to be minimized
    def minim(x):
        ketX = WPPrepare(x[0],x[1], input_state, dret1, dret2)        
        return -np.abs(ketX.T.conjugate() @ rho @ ketX)[0,0]
    #Start minimization from multiple initial guessses to avoid sub-optimal local extremes.
    Rs = [minimize(minim, g, bounds=[wp_bounds]*2, tol=tol) for g in search_grid_qh]
    #Pick global extereme.
    Fs = np.array([R['fun'] for R in Rs])
    idx = np.argmin(Fs)
    return Rs[idx]

def SearchForProj(ket, projector_state=np.array([[1],[0]]), dret1 = 0, dret2 = 2, tol=1e-6):
    """
    Search for wave plates angles x,y which makes projection onto desired
    ket state with given projector state with setup:
    input_state->QWP(y)-HWP(x)->
    --HWP(x)-QWP(y)->projector
    Args:
        ket - desired Jones vector to be projected on
        input_state - polarizers eigenstate
        dret1 - retardance error of half-wave plate
        dret2 - retardance error of quarter-wave plate
        tol - minimizer tolerance, see scipy.optimize.minimize docs.
    Returns:
        R - dictionary with minimization details. 
        R['x'] contains desired angles, R['fun'] is measure of quality (should be -1)
        See scipy.optimize.minimize docs for more details.
    """    
    #Construct function to be minimized
    def minim(x):
        bra = WPProj(x[0],x[1], projector_state, dret1, dret2)
        return -np.abs(bra @ ket)[0,0]**2
    #Start minimization from multiple initial guessses to avoid sub-optimal local extremes.
    Rs = [minimize(minim, g, bounds=[wp_bounds]*2, tol=tol) for g in search_grid_qh]
    #Pick global extereme.
    Fs = np.array([R['fun'] for R in Rs])
    idx = np.argmin(Fs)
    return Rs[idx]

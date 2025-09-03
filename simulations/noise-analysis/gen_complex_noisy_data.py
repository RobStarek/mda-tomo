import functools
import os
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib as mpl
import KetSugar as ks

SX = np.array([0,1,1,0]).reshape((2,2))
SZ = np.array([1,0,0,-1]).reshape((2,2))

rotations_tomo_proj = np.array((
    (0,0),
    (np.pi,0),
    (np.pi/2, 0),
    (np.pi/2, np.pi),
    (np.pi/2, 1*np.pi/2),
    (np.pi/2, 3*np.pi/2)
))

def apply_noise(rho_in, p_x=0.0, p_z=0.0): 
    rho_x = ks.ApplyOp(rho_in, SX)
    rho_bit_flipped = p_x*rho_x/2 + (1-p_x/2)*rho_in
    rho_phase_flipped = (1-p_z/2)*rho_bit_flipped + p_z*ks.ApplyOp(rho_bit_flipped, SZ)/2
    return rho_phase_flipped
    
def get_assumed_rpv(errors):
    coords = rotations_tomo_proj + errors.reshape((6,2))
    proj_kets = [ks.BlochKet(theta, phi) for theta, phi in coords]
    return np.array([ket @ ket.T.conjugate() for ket in proj_kets])

# def simulate_tomograms_pc(probes_coord, errors, counts=1_000_000):
#     tomograms = []
#     noise = 0.001
#     rpv_true = get_assumed_rpv(errors)
#     for theta, phi in probes_coord:
#         probe_ket = ks.BlochKet(theta, phi)
#         tomogram = np.array([ks.ExpectationValue(probe_ket, proj).real for proj in rpv_true])         
#         tomograms.append(tomogram*(1-noise) + noise)
#     tomograms = np.array(tomograms)
#     return np.random.poisson(tomograms*counts)

def simulate_tomograms_pc_noiseless(probes_coord, errors, noise_x=0.0, noise_y=0.0):
    tomograms = []
    noise = 0.001 #in the sense of overall uniform purity drop
    rpv_true = get_assumed_rpv(errors)
    for theta, phi in probes_coord:
        probe_ket = ks.BlochKet(theta, phi)
        probe_rho = ks.ketbra(probe_ket, probe_ket)
        probe_rho_noisy = apply_noise(probe_rho, noise_x, noise_y)                
        tomogram = np.array([ks.Overlap(probe_rho_noisy, proj).real for proj in rpv_true])         
        tomograms.append(tomogram*(1-noise) + noise)
    tomograms = np.array(tomograms)
    return tomograms
    
SAMPLES = 50

if __name__ == '__main__':
    probes = np.load('probes_samplings.npz')

    # truths = np.random.normal(0, 5, (SAMPLES,12))*np.pi/180.
    truths = np.load('inputs/test_data_n32_5sigma_0px_0pz.npz')['truths']
    # truths = np.load('test_data_n30_5sigma_0px_0pz').files
    
    #node 000
    test_data = np.array([
        simulate_tomograms_pc_noiseless(probes['n30'], true_deviation, noise_x=0, noise_y=0) for true_deviation in truths
        ])
    np.savez('test_data_n30_5sigma_0px_0pz.npz', truths = truths, test_data = test_data)

    #node 000 p
    test_data = np.array([
        simulate_tomograms_pc_noiseless(probes['n30'], true_deviation, noise_x=0.001, noise_y=0.001) for true_deviation in truths
        ])
    np.savez('test_data_n30_5sigma_1px_1pz.npz', truths = truths, test_data = test_data)

    #node 001p
    test_data = np.array([
        simulate_tomograms_pc_noiseless(probes['n30'], true_deviation, noise_x=0.005, noise_y=0.005) for true_deviation in truths
        ])
    np.savez('test_data_n30_5sigma_5px_5pz.npz', truths = truths, test_data = test_data)       

    #node 002
    test_data = np.array([
        simulate_tomograms_pc_noiseless(probes['n30'], true_deviation, noise_x=0.00, noise_y=0.01) for true_deviation in truths
        ])
    np.savez('test_data_n30_5sigma_0px_10pz.npz', truths = truths, test_data = test_data)    

    #node 002p
    test_data = np.array([
        simulate_tomograms_pc_noiseless(probes['n30'], true_deviation, noise_x=0.01, noise_y=0.01) for true_deviation in truths
        ])
    np.savez('test_data_n30_5sigma_10px_10pz.npz', truths = truths, test_data = test_data)

    #node 003
    test_data = np.array([
        simulate_tomograms_pc_noiseless(probes['n30'], true_deviation, noise_x=0.1, noise_y=0.1) for true_deviation in truths
        ])
    np.savez('test_data_n30_5sigma_100px_100pz.npz', truths = truths, test_data = test_data)    
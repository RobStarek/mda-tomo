import functools
import os
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib as mpl
import KetSugar as ks


rotations_tomo_proj = np.array((
    (0,0),
    (np.pi,0),
    (np.pi/2, 0),
    (np.pi/2, np.pi),
    (np.pi/2, 1*np.pi/2),
    (np.pi/2, 3*np.pi/2)
))

def get_assumed_rpv(errors):
    coords = rotations_tomo_proj + errors.reshape((6,2))
    proj_kets = [ks.BlochKet(theta, phi) for theta, phi in coords]
    return np.array([ket @ ket.T.conjugate() for ket in proj_kets])

def simulate_tomograms_pc(probes_coord, errors, counts=1_000_000):
    tomograms = []
    noise = 0.001
    rpv_true = get_assumed_rpv(errors)
    for theta, phi in probes_coord:
        probe_ket = ks.BlochKet(theta, phi)
        tomogram = np.array([ks.ExpectationValue(probe_ket, proj).real for proj in rpv_true])         
        tomograms.append(tomogram*(1-noise) + noise)
    tomograms = np.array(tomograms)
    return np.random.poisson(tomograms*counts)

def simulate_tomograms_pc_noiseless(probes_coord, errors):
    tomograms = []
    noise = 0.001
    rpv_true = get_assumed_rpv(errors)
    for theta, phi in probes_coord:
        probe_ket = ks.BlochKet(theta, phi)
        tomogram = np.array([ks.ExpectationValue(probe_ket, proj).real for proj in rpv_true])         
        tomograms.append(tomogram*(1-noise) + noise)
    tomograms = np.array(tomograms)
    return tomograms
    
SAMPLES = 50

if __name__ == '__main__':
    probes = np.load('probes_samplings.npz')

    truths = np.random.normal(0, 5, (SAMPLES,12))*np.pi/180.
    
    test_data = np.array([
        simulate_tomograms_pc(probes['n30'], true_deviation, counts = 1_000_000) for true_deviation in truths
        ])
    np.savez('test_data_n30_5sigma_1M.npz', truths = truths, test_data = test_data)

        
    test_data = np.array([
        simulate_tomograms_pc(probes['n30'], true_deviation, counts = 100_000) for true_deviation in truths
        ])
    np.savez('test_data_n30_5sigma_100k.npz', truths = truths, test_data = test_data)

    
    test_data = np.array([
        simulate_tomograms_pc(probes['n30'], true_deviation, counts = 10_000) for true_deviation in truths
        ])
    np.savez('test_data_n30_5sigma_10k.npz', truths = truths, test_data = test_data) 

    
    test_data = np.array([
        simulate_tomograms_pc(probes['n30'], true_deviation, counts = 1000) for true_deviation in truths
        ])
    np.savez('test_data_n30_5sigma_1k.npz', truths = truths, test_data = test_data)    

    
    test_data = np.array([
        simulate_tomograms_pc(probes['n30'], true_deviation, counts = 100) for true_deviation in truths
        ])
    np.savez('test_data_n30_5sigma_100.npz', truths = truths, test_data = test_data)    

    test_data = np.array([
        simulate_tomograms_pc_noiseless(probes['n30'], true_deviation) for true_deviation in truths
        ])
    np.savez('test_data_n30_5sigma_inf.npz', truths = truths, test_data = test_data)     
    print('done')


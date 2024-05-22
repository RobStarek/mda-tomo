import functools
import os
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib as mpl
import KetSugar as ks

# true_deviation = np.array([ 0.12981773,  0.05106024,  0.02192917,  0.0487319 , -0.10664317,
#         0.11696686,  0.00377647,  0.14331222,  0.14372837,  0.03825661,
#         0.14579775,  0.15912494])

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

def simulate_tomograms_pc(probes_coord, errors):
    tomograms = []
    noise = 0.001
    rpv_true = get_assumed_rpv(errors)
    for theta, phi in probes_coord:
        probe_ket = ks.BlochKet(theta, phi)
        tomogram = np.array([ks.ExpectationValue(probe_ket, proj).real for proj in rpv_true])
        tomograms.append(tomogram*(1-noise) + noise)
    return np.array(tomograms)

SAMPLES = 100

if __name__ == '__main__':
    probes = np.load('probes_samplings_x.npz')

    truths = np.random.normal(0, 5, (SAMPLES,12))*np.pi/180.
    test_data = np.array([
        simulate_tomograms_pc(probes['n32'], true_deviation) for true_deviation in truths
        ])
    ## uncomment this when you have backup of the original data
    #np.savez('test_data_n30_5sigma.npz', truths = truths, test_data = test_data)
    print('done')

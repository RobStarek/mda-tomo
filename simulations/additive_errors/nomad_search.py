import functools
import os
import pickle
from time import time
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib as mpl
import KetSugar as ks
import MaxLik as ml
import nomadlad
import concurrent.futures

SAMPLING_KEY = 'n30'
ML_ITERS = 10000
ML_THR = 1e-12

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
    return tomograms

probes = np.load('probes_samplings_x.npz')

def general_cost_f(x, tomograms = None):    
    rpv = get_assumed_rpv(x)
    rho_gen = (ml.Reconstruct(tomogram, rpv, ML_ITERS, 1e-9, RhoPiVect = True, Renorm = True) for tomogram in tomograms)
    purities = np.array([ks.Purity(rho).real for rho in rho_gen])
    return np.ptp(purities)

def get_cost_function(tomograms):
    return functools.partial(general_cost_f, tomograms = tomograms)

x0 = np.zeros(12)
x0_str = '  '.join(f'{x:.3f}' for x in x0)
bnds_up = '  '.join(['0.5']*12)
bnds_do = '  '.join(['-0.5']*12)

parameters = [
    'BB_OUTPUT_TYPE OBJ',
    'BB_MAX_BLOCK_SIZE 24',
    'MAX_BB_EVAL 100000',
    'SEED {}'.format(np.random.randint(0, 2 ** 31)),
    'ADD_SEED_TO_FILE_NAMES no',
    'DIMENSION 12',
    'DIRECTION_TYPE ORTHO 2N',
    f'LOWER_BOUND ( {bnds_do} )',
    f'UPPER_BOUND ( {bnds_up} )',
    'DISPLAY_DEGREE 0',
    'LH_SEARCH 1100 1100'
]

def gen_nomad_blackbox(point, tomogram = None):
    fw = get_cost_function(tomogram)(point)
    success = 1
    include = 1
    outcome = '{:.16f}'.format(fw)
    return success, include, outcome


test_data = np.load('test_data_n30_5sigma.npz')
test_tomograms = test_data['test_data']
if not('foos.pickle' in os.listdir()):
    cost_functions = []
    for i, test_tomogram in enumerate(test_tomograms):
        print(i, test_tomogram.shape)
        #cost_function = 
        cost_functions.append(functools.partial(gen_nomad_blackbox, tomogram = test_tomogram))
    with open('foos.pickle', 'wb') as jar:
        pickle.dump(cost_functions, jar)
else:
    with open('foos.pickle', 'rb') as jar:
        cost_functions = pickle.load(jar)

if __name__ == '__main__':
    ## underscore is ther to protect the original data results_n30.npy    
    with open('_results_n30.txt', 'w') as result_file:
        for i, blackbox in enumerate(cost_functions):
            print(f"--- {i} ---")
            t_start = time()            
            # Perform the optimization ritual.
            #with mpi4py.futures.MPIPoolExecutor() as executor: 
            with concurrent.futures.ProcessPoolExecutor(24) as executor:
                evaluator = functools.partial(executor.map, blackbox)
                result = nomadlad.minimize(evaluator, parameters)
            t_stop = time()
            flag, eval_count, best_feasible, best_infeasible = result            
            print(t_stop - t_start, "sec")
            f_true = blackbox(test_data['truths'][i])    
            print('truth f:', f_true, 'optimum f: ', best_feasible[0])            
            print('blackbox evaluation count        ', eval_count)
            print('final termination flag           ', flag)            
            li = [f'{i:02d}', f'{best_feasible[0]:.7f}']+[f'{x:.6f}' for x in best_feasible[1]]
            result_file.write('\t'.join(li)+'\n')
            result_file.flush()
    print("Done")



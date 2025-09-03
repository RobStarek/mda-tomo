import itertools
from time import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import KetSugar as ks
import MaxLik as ml
import pickle
from concurrent.futures import ProcessPoolExecutor
from functools import partial



butterfly = lambda x: ks.ketbra(x, x)
SHOTS = int(1e6)
ML_ITERS = 100
ML_THRES = 1e-9
DEG = np.pi/180.
SAMPLES = 500
SAMPLES_6 = 100


rotations_tomo_proj = np.array((
    (0,0),
    (np.pi,0),
    (np.pi/2, 0),
    (np.pi/2, np.pi),
    (np.pi/2, 1*np.pi/2),
    (np.pi/2, 3*np.pi/2)
))

def get_assumed_rpv(errors, return_ket=False):
    coords = rotations_tomo_proj + errors.reshape((6,2))
    proj_kets = [ks.BlochKet(theta, phi) for theta, phi in coords]
    if return_ket:
        return np.array(proj_kets)
    return np.array([ket @ ket.T.conjugate() for ket in proj_kets])

def haar_random_ket(Nqubits):
    """
    Generate a Haar-random matrix using the QR decomposition.
    Source: https://pennylane.ai/qml/demos/tutorial_haar_measure
    Thank you Penny Lane! 
    """
    dim = (1 << Nqubits)
    ket0 = np.zeros((dim,1))
    ket0[0,0] = 1
    re, im = np.random.normal(size=(dim, dim)), np.random.normal(size=(dim, dim))
    Z = re + 1j * im
    Q, R = np.linalg.qr(Z)
    # Step 3
    Lambda = np.diag([R[i, i] / np.abs(R[i, i]) for i in range(dim)])
    # Step 4
    return np.dot(Q, Lambda) @ ket0

#randomly_pick error but keep it fixed
# print(repr(np.round(np.random.normal(0, 5, 6*2),3)))
# I got these values
deltas = np.array([ 6.347, -9.125, -4.671, -1.102,  0.373, -8.484, -1.124, -3.204,
       -6.585,  9.503,  9.799, -3.137])*DEG

projs_true_q1 = get_assumed_rpv(deltas, return_ket=True)
projs_naive_q1 = get_assumed_rpv(deltas*0, return_ket=True)



def get_higher_projs(qubits, projs):
    return np.array([ks.kron(*kets) for kets in itertools.product(projs, repeat=qubits)])

def projs_to_rpv(projs):
    return np.array([butterfly(ket) for ket in projs])

QUBITS = [1, 2, 3, 4, 5, 6]
purs_collection = {}


edges = np.linspace(0.7,1, 30*2+1)


for qubits in QUBITS:
    print(f'--- {qubits} qubits ---')
    print("Getting projectors...")
    projs_true = get_higher_projs(qubits, projs_true_q1)
    dim = projs_true.shape[1]
    rpv_nai = projs_to_rpv(get_higher_projs(qubits, projs_naive_q1))
    rpv_tru = projs_to_rpv(projs_true)

    _SAMPLES = SAMPLES
    if qubits>=6:
        _SAMPLES = SAMPLES_6
    
    print("Getting Haar-kets...")
    kets = np.array([haar_random_ket(qubits) for i in range(_SAMPLES)]).reshape((_SAMPLES, dim))

    print("Generating data...")
    tomos = np.abs(np.einsum('ik,jk->ij', kets, projs_true.reshape((-1,dim)).conj()))**2 + 1e-6

    # tomos = (projs_true.conj().reshape((-1,dim)) @ kets.reshape((SAMPLES, -1)).T).T

    print("Maxlik...")
    # #recostrunct the samples
    reconstruct_foo_nai = partial(ml.Reconstruct, RPVket = rpv_nai, max_iters=ML_ITERS, tres=ML_THRES, RhoPiVect=True, Renorm=True)
    reconstruct_foo_tru = partial(ml.Reconstruct, RPVket = rpv_tru, max_iters=ML_ITERS, tres=ML_THRES, RhoPiVect=True, Renorm=True)

    chunk = int(_SAMPLES//20)
    with ProcessPoolExecutor(20) as executor:
       naive_rec = np.array(list(tqdm(executor.map(reconstruct_foo_nai, tomos, chunksize=chunk), total=_SAMPLES)))
       tru_rec = np.array(list(tqdm(executor.map(reconstruct_foo_tru, tomos, chunksize=chunk), total=_SAMPLES)))

    np.savez(f'saved_samples_q{qubits}.npz', naive=naive_rec, calib=tru_rec, reference=kets)
    # naive_rec = np.array([ml.Reconstruct(tomo, rpv_nai, ML_ITERS, ML_THRES, RhoPiVect=True, Renorm=False) for tomo in tqdm(tomos)])
    # tru_rec = np.array([ml.Reconstruct(tomo, rpv_tru, ML_ITERS, ML_THRES, RhoPiVect=True, Renorm=True) for tomo in tqdm(tomos)])

    print("Purities...")
    purs_collection[f'q{qubits}_nai'] = np.einsum('kij,kji->k',naive_rec, naive_rec)
    purs_collection[f'q{qubits}_tru'] = np.einsum('kij,kji->k',tru_rec, tru_rec)

    print("Saving...")
    with open('temp_jar2.pickle', 'wb') as mf:
        pickle.dump(purs_collection, mf)
    
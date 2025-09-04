# -*- coding: utf-8 -*-
from re import A
import numpy as np
from scipy.optimize import minimize
import KetSugar as ks
import departed #https://github.com/jan-provaznik/departed


"""
Little toolbox for manipulating unitary matrices and Choi matrices.
Dependencies: numpy, scipy, KetSugar

References:
M.-D. Choi, Completely Positive Linear Maps on Complex Matrices, Linear Algebra and its Applications, 10, 285–290 (1975).
https://en.wikipedia.org/wiki/Channel-state_duality
"""

def ChiKetToKraus(chiket):
    """
    Fold n-qubit process ket (in Choi-Jamilkowski formalism) into operator matrix.
    Args:
        chiket ... column ndarray (2^(2n) x 1), complex
    Returns:
        U ... operator ndarray (2^n x 2^n)
    """
    norm = ks.braket(chiket, chiket)
    d = chiket.shape[0]
    dim = int(d**0.5)    
    U = chiket.ravel().reshape((dim, dim)).T
    U = norm*U*(dim**0.5)
    return U

def GetKrausFromChi(chi):
    """
    Decompose Choi matrix of a n-qubit process into its Kraus eigenoperators
    Args:
        chi ... column ndarray (2^(2n) x 2^(2n)), complex
    Returns:
        eigevalues ... corresponding probabilities of operators
        matrices ... list of 2^(2n) operators, ndarrays (2^n x 2^n)
    """    
    eigenvalues, eigenkets = np.linalg.eigh(chi)
    d = chi.shape[0]
    matrices = [ChiKetToKraus(eigenkets[:,i].reshape((d,1))) for i in range(d)]
    return eigenvalues, matrices

def GuessUfromChoi(Chi):
    """
    Guess unitary parameters from Choi matrix Chi. 
    Chi matrix has to origin in unitary operation for this to work well.
    Returns:
        theta, phi1, phi2 - real-valued unitary parameters
    """
    if Chi[0,0] == 0:
        theta = np.pi/2
        phi1 = 0
        phi2 = 0
        return theta, phi1, phi2
    elif Chi[1,1] ==0:
        theta = 0
        phi1 = 0
        phi2 = 0
        return theta, phi1, phi2
    theta = np.arctan((Chi[1,1]/Chi[0,0])**.5)
    phi1 = np.angle(Chi[0,3]/Chi[0,0])*0.5
    phi2 = np.angle(Chi[1,2]/Chi[1,1])*0.5
    return theta, phi1, phi2

def UtoChi(U, ket=False):
    """
    Transform single-qubit unitary matrix U to Choi matrix Chi 
    or Choi ket (when ket = True).
    """
    Bell = np.array([[1],[0],[0],[1]])/2**.5
    ChiKet = ks.kron(np.eye(2),U) @ Bell
    if ket:
        return ChiKet
    Chi = ks.ketbra(ChiKet, ChiKet)
    return Chi


def UtoChiD(U, ket=False):
    """
    Transform multi-qubit channel matrix into Choi ket/matrix.
    """
    n = U.shape[0]    
    Base = [ks.BinKet(i,n-1) for i in range(n)]
    Bell = sum([np.kron(ket, ket) for ket in Base])    
    ChiKet = np.kron(np.eye(n), U) @ Bell
    if ket:
        return ChiKet
    Chi = ks.ketbra(ChiKet, ChiKet)
    return Chi

def UfromParam(theta, phi1, phi2):
    """
    Construct single-qubit unitary matrix from given unitary parameters
    theta, phi1, phi2.    
    """
    alpha = np.cos(theta)*np.exp(1j*phi1)
    beta = np.sin(theta)*np.exp(1j*phi2)
    U = np.array([
        [alpha, -beta.conjugate()],
        [beta, alpha.conjugate()]
        ])
    return U

def ChiKetFromParam(theta, phi1, phi2):
    """
    Construct Choi ket from given unitary parameters.
    """
    U = UfromParam(theta, phi1, phi2)
    ChiKet = UtoChi(U, ket = True)
    return ChiKet

def FitChiU(Chi):
    """
    Find closest unitary to given Choi matrix.
    Returns:
        Result dictionary R, see scipy.optimize.minimize for more details.
    """
    def minim(x):
        ket = ChiKetFromParam(x[0], x[1], x[2])
        return -ks.ExpectationValue(ket, Chi).real
    
    guess = GuessUfromChoi(Chi)
    R = minimize(minim, guess)
    return R    

def MapTransform(rho, chi, renorm = True):
    """
    Transform input density matrix with Choi matrix.
    Tr_i[(1 \otimes RhoIn.T) chi (...)^\dagger]
    Args:
        rho - density matrix to be transformed
        chi - quantum process matrix
        renorm - (default true), trace-normalize output
    Returns:
        transformed density matrix
    """
    d = rho.shape[0] #number of rows
    n = int(np.log2(d)) #number of qubits
    trace_list = [1]*n + [0]*n

    POVM = np.kron(rho.T, np.eye(d, dtype=complex))
    ChiTrans = POVM @ chi #@ ks.dagger(POVM)
    #RhoOut = ks.TraceOverQubits(ChiTrans, trace_list)
    RhoOut = departed.ptrace(ChiTrans, [2]*(2*n), trace_list) 
    if renorm:
        return RhoOut/np.trace(RhoOut)
    else:
        return RhoOut

def ChainTwoChoisSym(choi_a, choi_b):
    """
    Contatenate two Choi matrices of same dimensions.
    S stands for symmetrical version (input and output spaces have the same dimension)        
    """
    d = choi_a.shape[0]
    n = int(np.log2(d)) #number of qubits in choi matrix
    nhalf = int(n//2) #number of qubits in input space
    component_dims = [2]*n
    component_dims_2 = [2]*(n+nhalf)

    mask = [1]*nhalf + [0]*nhalf
    mask_2 = [0]*nhalf + [1]*nhalf + [0]*nhalf

    eye = np.eye(2**nhalf)
    ptA = np.kron(choi_a, eye)
    chi_b_t1 = departed.ptranspose(choi_b, component_dims, mask)
    ptB = np.kron(eye, chi_b_t1)

    return departed.ptrace(ptA @ ptB, component_dims_2, mask_2)


def ChainTwoChoisAsym(choi_a, choi_b, a_dims, input_a_mask, b_dims, input_b_mask):    
    """
    Contatenate two Choi matrices of same dimensions.
    The resulting operation is equivalent of first applying channel A and then channel B.
    A stands for asymmetrical version, where all matrices does not have to have the same
    dimensions. But it still must be that dim(choi_a_output) == dim(choi_b_input).

    It uses formula
    :math:`\chi_{a,01} \circ \chi_{b,12} = \mathrm{Tr}_{1}[(\chi_{a,01} \otimes 1_{2})\cdot (1_0 \otimes \chi_{b,12}^{T_1})]`
    
    Args:
        choi_a, choi_b ... Choi matrices to be concatenated
        a_dims ... list of system dimensions of the choi_a
        input_a_mask ... list with 1 (True) at places where the system represents input space
           warning: it should be aligned, i.e. [1,1,0,0] is OK, but [0,1,0,1] not
        b_dims, input_b_mask ... analogous for choi_b
    Returns:
        choi_a o choi_b ... concatenated proces matrix
        out_mask ... list where elements are 1 if the sub-system represents input space
        out_dims ... list with dimensions of the subsystems    
    """        
    arr_a_dims = np.array(a_dims)
    arr_b_dims = np.array(b_dims)
    arr_a_mask = np.array(input_a_mask).astype(bool)
    arr_b_mask = np.array(input_b_mask).astype(bool)
    arr_bo_mask = np.invert(arr_b_mask)
    
    eye_ai= np.eye(np.prod(arr_a_dims[arr_a_mask]))
    eye_bo= np.eye(np.prod(arr_b_dims[arr_bo_mask]))    
    dims = np.concatenate([arr_a_dims[arr_a_mask], arr_b_dims])
    transpose_mask = np.concatenate([arr_a_mask[arr_a_mask]*False, arr_b_mask])
    ptA = np.kron(choi_a, eye_bo)
    chi_b_t1 = departed.ptranspose(choi_b, b_dims, input_b_mask)
    ptB = np.kron(eye_ai, chi_b_t1)      
    out_mask = list(np.concatenate([arr_a_mask[arr_a_mask], arr_b_mask[arr_bo_mask]]))
    out_dims = list(np.concatenate([arr_a_dims[arr_a_mask], arr_b_dims[arr_bo_mask]]))
    return departed.ptrace(ptA @ ptB, dims, transpose_mask), list(out_mask), list(out_dims)


def ChainChoisAsym(choi_list, dims_list, input_mask_list):
    """
    Chain (contatenete) multiple Choi matrices in order of their appearence in the list.

    Args:
        choi_list ... list with choi matrices to be concatenated
        dim_list ... list of lists containing subsystem dimensions
        input_mask_list ... list of lists containing '1' if the subsystem represents input space
    Returns:
        choi_eff ... concantenated process matrix
    """    
    choi_eff = None
    dims_eff = None
    mask_eff = None
    for i, (choi, dims, mask) in enumerate(zip(choi_list, dims_list, input_mask_list)):
        if i == 0:
            choi_eff = choi
            dims_eff = dims
            mask_eff = mask
            continue
        #print(dims_eff, ":", dims)
        #print(mask_eff, ":", mask)
        choi_eff, mask_eff, dims_eff = ChainTwoChoisAsym(
            choi_eff, choi, dims_eff, mask_eff, dims, mask
        )
    return choi_eff

# -*- coding: utf-8 -*-
"""MaxLik.py
Discrete-variable quantum maximum-likelihood reconstruction.
V1.1

This module provides a simple numpy-implementation of Maximum likelihood reconstruction
method [1,2] for reconstructing low-dimensional quantum states and processes (<=6 qubits in total).

This package is limited to projection and preparation of pure states.

An orderd list of prepared/projected list is provided to MakeRPV() function to
generate auxiliary array of projection-preparation matrices (Rho-Pi vector). 

The Rho-Pi vector is inserted as an argument together with data to Reconstruct() function.
The Reconstruct() function returns reconstructed density matrix.

Example:
    Minimal single-qubit reconstruction
        import numpy as np
        from MaxLikCore import MakeRPV, Reconstruct
        #Definition of projection vector
        LO = np.array([[1],[0]])
        HI = np.array([[0],[1]])
        Plus = (LO+HI)*(2**-.5)
        Minus = (LO-HI)*(2**-.5)
        RPlu = (LO+1j*HI)*(2**-.5)
        RMin = (LO-1j*HI)*(2**-.5)
        #Definion of measurement order, matching data order
        Order = [[LO,HI,Plus,Minus,RPlu,RMin]]
        #Measured counts
        testdata = np.array([500,500,500,500,1000,1])
        #Prepare (Rho)-Pi vect
        RPV = MakeRPV(Order, False)
        #Run reconstruction
        E = Reconstruct(testdata, RPV, 1000, 1e-6)

References:
    1. Fiurasek, Hradil, Maximum-likelihood estimation of quantum processes, 
       Phys. Rev. A 63, 020101(R) (2001) 
       https://journals.aps.org/pra/abstract/10.1103/PhysRevA.63.020101
    2. Paris (ed.), Rehacek, Quantum State Estimation - 2004, 
       Lecture Notes in Physics, ISBN: 978-3-540-44481-7, 
       https://doi.org/10.1007/b98673

Todo:
    * ?

"""

import itertools
from functools import reduce
import numpy as np

try:
    import numba

    allow_numba = True
except ImportError as e:
    allow_numba = False
allow_numba = True  # manual override, just for testing
print(
    "MaxLik: Numba Allowed:",
    allow_numba,
    "=> use",
    "cycle-based" if allow_numba else "vectorized",
    "K-vector construction",
)


def RPVketToRho(RPVKet, trans_and_flatten=False):
    """
    Convert ndarray of sorted preparation-detection kets into ndarray of
    density matrices.

    Args:
        RPVket: n x d ndarray (line-vectors) containing measured projections or preparation-projection vectors.
        or n x 1x d nd array (column vectors)

    Returns:
        RhoPiVect: n x d x d ndarray of density matrices made up from RPV kets.
    """
    n, dim, _ = RPVKet.shape
    view = RPVKet.reshape((n, dim)).T
    cview = view.conj()
    if trans_and_flatten:
        # for vectorized version,
        # do the transposition and flattening here
        return np.einsum("ij,kj->kij", view, cview).ravel().reshape((-1, n))
    return np.einsum("ij,kj->jik", view, cview).reshape((n, dim, dim))


ALPHABET = ",".join([chr(97 + i) for i in range(10)])


def veckron(*vecs):
    """Kronecker product of multiple vectors.
    If there is only 1 vector, result is trivial.
    For up to 9 vectors, it is implemented with einsum.
    For more vector, it falls back to reduce on np.kron
    Args:
        *vecs (ndarray) : column vectors to be multiplied
    Returns:
        (complex ndarray) : new column vector
    """
    n = len(vecs)
    if n == 1:
        return vecs[0]
    if n > 9:
        return reduce(np.kron, vecs, np.eye(1, dtype=complex))
    # EINSUM_EXPRS_VEC[len(vecs)-2],
    return np.einsum(ALPHABET[0 : 2 * n - 1], *(v.ravel() for v in vecs)).reshape(-1, 1)


def MakeRPV(Order, Proc=False):
    """
    Create list preparation-projection kets.
    Kets are stored as line-vectors or column-vectors in n x d ndarray.
    This function is here to avoid writing explicit nested loops for all combinations of measured projection.

    Args:
        Order: list of measured/prepared states on each qubit, first axis denotes qubit, second measured states, elements are kets stored as line-vectors or column-vectors.
        Proc: if True, first half of Order list is regarded as input states and therefore conjugated prior building RPV ket.

    Returns:
        RPVvectors: complex ndarray of preparation-projection kets, order should match the measurement
    """
    n_half = len(Order) / 2
    views = (
        map(np.conjugate, proj_kets) if ((i < n_half) and Proc) else proj_kets
        for i, proj_kets in enumerate(Order)
    )
    return np.array([veckron(*kets) for kets in itertools.product(*views)], dtype=complex)


def ReconstutionLoopVect(data, E, RhoPiVect, dim, max_iters, tres, projs, renorm=False):
    """
    Internal function for loop-based K-operator construction.
    When numba is not available, use this vectorized K-operator construction
    Inner loop of the Reconstruction method.
    It is separated in order to compile it with numba.
    Args:
        RPVAux: hstacked RPVket
        OutPiRho: hstacked RhoPiVect block-wise-multiplied with data
        max_iters, tres, renorm: see mother method.
    Returns:
        Reconstructed matrix E
    """
    count = 0
    meas = 10 * tres
    # iterate until you reach threshold in frob. meas. of diff or
    # exceed maximally allowed number of steps

    if renorm:
        # check
        # ProjSum = np.sum(RhoPiVect, axis=0) #Sum of projectors should be ideally 1
        ProjSum = (
            np.sum(RhoPiVect, axis=1).reshape((dim, dim)).T
        )  # Sum of projectors should be ideally 1

        Lam, U = np.linalg.eig(
            ProjSum
        )  # eigen-decomposition of the vectors to calculate square root of inverse matrix
        LamInv = np.diag(Lam**-1)
        Hsqrtinv = U @ LamInv @ U.T.conjugate()  # square root of inverted matrix

    while count < max_iters and meas > tres:
        Ep = E  # Original matrix for comparison
        Aux = (E.ravel().reshape((1, -1)) @ RhoPiVect).ravel().real
        factor = data / Aux  # multiply with data
        K = (RhoPiVect @ factor.reshape((-1, 1))).reshape(dim, dim).T
        if renorm:
            # factor2 = E.T.ravel() @ ProjSum.ravel()
            HS = Hsqrtinv  # *(factor2)
            E = HS @ K @ E @ K @ HS
        else:
            E = K @ E @ K
        E = E / np.trace(E)  # norm the matrix
        # Equivalent to np.linalg.norm, .i.e. Frob. norm
        meas_aux = (E - Ep).ravel()
        meas = (meas_aux @ meas_aux.conj()).real ** 0.5
        count += 1  # counter increment
    return E


def GrammSchmidt(X, row_vecs=False, norm=True):
    """
    Vectorized Gramm-Schmidt orthogonalization.
    Creates an orthonormal system of vectors spanning the same vector space
    which is spanned by vectors in matrix X.

    Args:
        X: matrix of vectors
        row_vecs: are vectors store as line vectors? (if not, then use column vectors)
        norm: normalize vector to unit size
    Returns:
        Y: matrix of orthogonalized vectors
    Source: https://gist.github.com/iizukak/1287876#gistcomment-1348649
    """
    if not row_vecs:
        X = X.T
    Y = X[0:1, :].copy()
    for i in range(1, X.shape[0]):
        proj = np.diag((X[i, :].dot(Y.T) / np.linalg.norm(Y, axis=1) ** 2).flat).dot(Y)
        Y = np.vstack((Y, X[i, :] - proj.sum(0)))
    if norm:
        Y = np.diag(1 / np.linalg.norm(Y, axis=1)).dot(Y)
    if row_vecs:
        return Y
    else:
        return Y.T


def _ReconstructLoopCycles(
    data, E, K, RhoPiVect, dim, max_iters, tres, projs, ProjSum, Hinv, renorm=False
):
    """
    Internal function for loop-based K-operator construction, written in numba.
    """
    count = 0
    meas = 10 * tres

    while count < max_iters and meas > tres:
        Ep = E  # Original matrix for comparison
        K[:, :] = 0  # reset K operator
        for i in range(projs):
            if data[i] == 0:
                continue
            Denom = RhoPiVect[i].T.ravel() @ E.ravel()
            K = K + RhoPiVect[i] * (data[i] / Denom)
        if renorm:
            # factor2 = E.T.ravel() @ ProjSum.ravel() #sum of p_j over j
            HS = Hinv  # *(factor2)
            E = HS @ K @ E @ K @ HS
        else:
            E = K @ E @ K
        TrE = np.sum(np.diag(E))
        E = E / TrE  # norm the matrix
        meas = abs(np.linalg.norm((Ep - E)))  # threshold check
        count += 1  # counter increment
    return E


# Define function ReconstructionLoop() and set numba jit, when possible
if allow_numba:
    try:
        ReconstutionLoopCycles = numba.jit(nopython=True)(_ReconstructLoopCycles)
    except Exception as e:
        ReconstutionLoopCycles = _ReconstructLoopCycles
        print(
            "MaxLik: Numba decoraration of inner loop function failed. The program migh be slower."
        )


def Reconstruct(data, RPVket, max_iters=100, tres=1e-6, **kwargs):
    """
    Maximum likelihood reconstruction of quantum state/process.

    Args:
        data: ndarray of n real numbers, measured in n projections of state.
        RPVket: Complex n x d (line vectors) or n x d x 1 (column vectors) ndarray of kets describing the states
        that the measured state is projected onto.
        max_iters: integer number of maximal iterations
        tres: when the change of estimated matrix measured by Frobenius norm is less than this, iteration is stopped
    Kwargs:
        RhoPiVect: explicit array of matrices with projectors, if entered, RPVket is passed to RhoPiVect
        Renorm: if True, projector renormalization will be done each round

    Returns:
        E: estimated density matrix, d x d complex ndarray.
    """
    dim = RPVket.shape[1]  # pylint might complain about this, but it is really OK
    projs = data.size

    if kwargs.get("RhoPiVect", False):
        if allow_numba:
            # for numba, just use RPVket argument withou further ado
            RhoPiVect = RPVket
        else:
            # arrange transposed projector matrices into columns
            RhoPiVect = np.einsum("ijk->kji", RPVket).reshape((dim * dim, projs))
    else:
        RhoPiVect = RPVketToRho(RPVket, not (allow_numba))
    renorm = kwargs.get("Renorm", False)

    E = np.zeros((dim, dim), dtype=complex)
    np.fill_diagonal(E, 1.0 / dim)

    if allow_numba:
        # Use cycle-based construction of K-operator, when numba is available.
        K = np.zeros((dim, dim), dtype=complex)
        H = np.zeros((dim, dim), dtype=complex)
        S = np.sum(RhoPiVect, axis=0)
        if renorm:
            Lam, U = np.linalg.eigh(S)
            if np.sum(np.abs(Lam - Lam[0])) < dim * 1e-14:
                U = np.eye(dim, dtype=complex)
            elif np.abs(np.sum(U @ U.T.conjugate() - np.eye(dim))) > dim * 1e-14:
                U = GrammSchmidt(U, False, True)
            LamInv = np.diag(Lam**-1)
            Hsqrtinv = U @ LamInv @ U.T.conjugate()
        else:
            Hsqrtinv = H

        return ReconstutionLoopCycles(
            data, E, K, RhoPiVect, dim, max_iters, tres, projs, S, Hsqrtinv, renorm
        )
    else:
        # Use vectorized contruction of K-operator
        return ReconstutionLoopVect(data, E, RhoPiVect, dim, max_iters, tres, projs, renorm)


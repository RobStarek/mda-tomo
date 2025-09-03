# -*- coding: utf-8 -*-
import numpy as np
import scipy.linalg
"""
Shorthand notation, basic constants and frequently used function
for some basic low-dimensional DV quantum mechanics.
"""

#Constants
LO = np.array([[1],[0]])
HI = np.array([[0],[1]])
HLO = (LO+HI)*(2**-.5)
HHI = (LO-HI)*(2**-.5)
CLO = (LO+1j*HI)*(2**-.5)
CHI = (LO-1j*HI)*(2**-.5)

#Short-hand notation
def dagger(x : np.ndarray):
    """
    Hermite conjugation of x.
    """
    return x.T.conjugate()

def braket(x : np.ndarray, y : np.ndarray):
    """
    Inner product of two ket-vectors -> C-number
    """
    return np.dot(x.T.conjugate(), y)[0,0]

def ketbra(x : np.ndarray, y : np.ndarray):
    """
    Outer product of two ket-vectors -> C-matrix
    """
    return np.dot(x, y.T.conjugate())

def kron(*arrays):
    """
    Multiple Kronecker (tensor) product.
    Multiplication is performed from left.    
    """
    E = np.eye(1, dtype=complex)
    for M in arrays:
        E = np.kron(E,M)
    return E

def BinKet(i=0,imx=1):
    """
    Computational base states i in imx+1-dimensional vectors.
    """
    ket = np.zeros((imx+1,1), dtype=complex)
    ket[i] = 1
    return ket

#Ket constructors
def BlochKet(theta, phi):
    """
    Return a qubit column-vector with Bloch sphere coordinates theta, phi.
    theta - lattitude measured from |0> on Bloch sphere
    phi - longitude measured from |+> on Bloch sphere (phase)
    """
    return np.array([[np.cos(theta/2)], [np.sin(theta/2)*np.exp(1j*phi)]])

#Routinely used simple functions
def Overlap(MA, MB):
    """
    Normalized overlap of two density matrices MA, MB.
    When at least one of the matrices is pure, it is equivalent to fidelity.
    """
    #equivalent to np.trace(MA @ MB)/(np.trace(MA)*np.trace(MB)), but off-diagonal elements are not computed
    return (MA.T.ravel() @ MB.ravel())/(np.trace(MA)*np.trace(MB))

def Purity(M):
    """
    Purity of the density matrix M.
    For n qubits, minimum is (2^n).
    """
    norm = np.trace(M)
    #equivalent to np.trace(M @ M)/(norm**2), but off-diagonal elements are not computed
    return (M.T.ravel() @ M.ravel())/(norm**2)

def ApplyOp(Rho,M):
    """
    Calculate M.Rho.dagger(M).
    """
    return M @ Rho @ M.T.conjugate()

def ExpectationValue(Ket, M):
    """
    Expectation value <bra|M|ket>.
    """
    return (Ket.T.conjugate() @ M @ Ket)[0,0]

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
        proj = np.diag(
            (X[i, :].dot(Y.T.conj())/np.linalg.norm(Y, axis=1)**2).flat).dot(Y)
        Y = np.vstack((Y, X[i, :] - proj.sum(0)))
    if norm:
        Y = np.diag(1/np.linalg.norm(Y, axis=1)).dot(Y)
    if row_vecs:
        return Y
    else:
        return Y.T

def sqrtm(M):
    """
    Matrix square root of density matrix M.
    Calculation is based on eigen-decomposition.    
    """
    Di, Rot = np.linalg.eig(M)
    rank = np.sum((np.abs(Di) > 2*np.finfo(float).eps))
    Di = np.sqrt(Di)
    Di[np.isnan(Di)] = 0
    Di = np.diag(Di)
    if (rank == M.shape[0]):
        #Full rank => Hermitean transposition is actually inversion
        RotInv = Rot.T.conjugate()
    elif rank == 1:
        #Rank 1 => The state is pure and the matrix is it's own square-root.
        return M
    else:
        #(1 < Rank < dimension) => at least one eigenvalue is zero, orthogonalize found unitary
        #in order to perform Hermitean inversion of the rotation matrix
        #If this was not the case, zero eigenvalue can correspond to 
        #arbitrary vector which would destroy unitarity of Rot matrix.
        RotGs = GrammSchmidt(Rot, False, True)
        RotInv = RotGs.T.conjugate()
    N = np.dot(np.dot(Rot, Di), RotInv)
    return N

def Fidelity(A, B):
    """
    Fidelity between density matrices A, B.
    Accepts both mixed. If A or B is pure, consider using Overlap().
    """
    #A0 = sqrtm(A)
    Ax = A/np.trace(A)
    Bx = B/np.trace(B)
    A0 = scipy.linalg.sqrtm(Ax)
    #use scipy to do sqrtm, when it fails (for singular matrices), try KetSugars sqrtm method
    if np.any(np.isnan(A0)):
        A0 = sqrtm(Ax)    
    A1 = (np.dot(np.dot(A0, Bx), A0))
    A2 = scipy.linalg.sqrtm(A1)
    if np.any(np.isnan(A2)):
        A2 = sqrtm(A1)       
    return np.abs(A2.trace())**2

def TraceLeft(M):
    """
    Partial trace over left-most qubit.
    """
    n = M.shape[0]
    return M[0:n//2, 0:n//2] + M[n//2:n, n//2:n]

def TraceRight(M):
    """
    Partial trace over right-most qubit.
    """
    n = M.shape[0]
    blocks = n//2
    TrM = np.zeros((blocks, blocks), dtype=complex)
    for i in range(blocks):
        for j in range(blocks):
            TrM[i,j] = np.trace(M[i*2:(1+i)*2, j*2:(1+j)*2])
    return TrM

def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size
    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    
    Source: https://stackoverflow.com/a/16873755
    """
    h = arr.shape[0]
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1, 2)
               .reshape(-1, nrows, ncols))

def TraceMiddle(M):
    """
    Partial trace, trace middle qubit of the 3-qubit density matrix
    """
    blocks = blockshaped(M, 2,2)
    sum_coord = [[0,5],[2,7],[8,13],[10,15]]
    
    temp = []
    for idx1, idx2 in sum_coord:
        temp.append(blocks[idx1]+blocks[idx2])
    
    new_arr = np.zeros((4,4),dtype=complex)
    new_arr[0:2, 0:2] = temp[0]
    new_arr[0:2, 2:4] = temp[1]
    new_arr[2:4, 0:2] = temp[2]
    new_arr[2:4, 2:4] = temp[3]
    return new_arr
    


#Helper function for TraceOverQubits()
def expandnum(number, mask, bits):
    """
    Expand number bitwise to mask, given the number of bits.
    Example:
    expandnum(0b10, 0b0101, 4) -> 0b0100
    """
    s = 0
    k = 0
    for j in range(bits):
        mbit = (mask >> j) & 0b1
        vbit = (number >> k) & 0b1
        s += ((vbit*mbit) << j)
        k += int(mbit)
    return s

def TraceOverQubits(M,li):
    """
    General partial trace of square matrix M over qubits specified in li.
    It is bit slower than TraceLeft functions due to internal cycles.

    Args:
        M ... square matrix with power of 2 dimension
        li ... list with as many elements as qubits. If element is 1, then the qubit is traced over.
           if element is 0, qubit is kept.
    Returns:
        Mnew ... matrix after partial trace
    Raises:
        "Trace list does not match the matrix." when the dimension of the matrix does
        not match the length of list of qubits.
    """
    if len(li) != np.log2(M.shape[0]):
        raise("Trace list does not match the matrix.")
    mask = 0
    negmask = 0
    for i, k in enumerate(li[::-1]):
        mask += int(k)*(2**i)
        negmask += int(not(k))*(2**i)    
    bits = len(li) #number of qubits in M
    nbits = sum(li) #number of traced-out qubits
    ndim = 2**(bits-nbits) #dimension of result matrix
    sumdim = 2**nbits #how many elements we sum
    MS = np.zeros((ndim, ndim), dtype=M.dtype)
    for i in range(ndim):
        for j in range(ndim):
            i0 = expandnum(i, negmask, bits)
            j0 = expandnum(j, negmask, bits)
            for k in range(sumdim):
                k0 = expandnum(k, mask, bits)
                MS[i,j] = MS[i,j] + M[i0+k0,j0+k0]
    return MS

def PartialTranspose(M, li):
    """
    Generan n-qubit partial transposition.
    Args:
        M ... density matrix to transposed
        li ... list with specification which qubits to be transposed
               1 when qubit is to be transposed, 0 otherwise
    Returns:
        partially transposed matrix
    Raises:
        "Trace list does not match the matrix." when the dimension of the matrix does
        not match the length of list of qubits.
    Example:
        RhoABTC = PartialTranspose(RhoABC, [0,1,0])
    """
    dim = M.shape[0]
    if len(li) != np.log2(dim):
        raise("Trace list does not match the matrix.") 

    #make transposition mask
    mask = 0
    for i, k in enumerate(li[::-1]):
        mask += k*(2**i)
    
    MT = np.zeros_like(M)
    #this could be optimized for Hermitian matrices
    for i in range(dim):
        for j in range(dim):
            di = i & mask
            dj = j & mask
            deltaij = di - dj
            MT[i - deltaij, j + deltaij] = M[i,j]
    return MT

def ConcurrenceAlt(M):
    """
    Alternative formulation of a concurrence.
    see https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.80.2245
    Instead of performind two square roots, just use square root of 
    eigenvalues of M . spin_flipped_M.
    """
    flip_op = np.array([[0,0,0,-1],[0,0,1,0],[0,1,0,0],[-1,0,0,0]])
    RhoFlip = flip_op @ M.conjugate() @ flip_op            
    R = M @ RhoFlip   
    eigs = np.linalg.eigvals(R)
    eigs = np.sort(eigs) #ascending order
    number1 = eigs[3]**.5 - (eigs[2]**.5+eigs[1]**.5+eigs[0]**.5)
    return max(0, number1)

def EntOfForm(rho):
    """
    Entanglement of formation of two-qubit density matrix.
    see https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.80.2245
    """
    C = ConcurrenceAlt(rho)
    x_arg = (1 + (1-C*C)**.5)*0.5
    if x_arg == 0 or x_arg == 1:
        return 0
    else:
        return (-x_arg*np.log2(x_arg) - (1-x_arg)*np.log2(1-x_arg))

def VonNeumannEntropy(rho, ln=False):
    """
    Von Neumann Entropy of a density matrix.
    Log2 is taken in the context of information (ln = False).
    Natural log is taken in the context of themodynamics (ln = True)
    S = -Sum_i [lambda_i log2(lambda_1)]
    Defined in Nielsen, Michael A. and Isaac Chuang (2001). Quantum computation and quantum information
    or on wiki: https://en.wikipedia.org/wiki/Von_Neumann_entropy
    Args:
        rho ... density matrix (Hermitian)
        base ... 2 for log2, 
    Returns: 
        real number
    """
    eigs = np.linalg.eigh(rho)[0]
    nzeig = eigs[eigs > 0] #non-zero eigenvalues    
    if ln:
        return -1*(nzeig @ np.log(nzeig)) #sum of products
    else:    
        return -1*(nzeig @ np.log2(nzeig)) #sum of products

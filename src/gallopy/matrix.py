import numpy as np
from numpy.fft import fftshift, fftn
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from typing import Union, Callable, Sequence
from numbers import Number

def convmat(A, P, Q):
    # Extract spatial harmonics (P, Q, R) of a general 3D unit cell
    
    # Extract shape of an array
    
    Nx, Ny = np.shape(A)
    
    # Spatial harmonic indices
    
    NH = P * Q  # Total num of spatial harmonics
    
    p = np.array(np.arange(-np.floor(P / 2), np.floor(P / 2) + 1))  # Idx in x dir
    q = np.array(np.arange(-np.floor(Q / 2), np.floor(Q / 2) + 1))  # Idx in y dir
    
    # Array indices for the zeroth harmonic
    
    p_0 = int(np.floor(Nx / 2))  # add +1 in matlab
    
    q_0 = int(np.floor(Ny / 2))
    
    # Fourier coefficients of A
    
    A = fftshift(fftn(A) / (Nx * Ny))  # Ordered Fourier coeffs
    
    # Init Convolution matrix;
    
    C = np.zeros((NH, NH), dtype='complex')
    
    # Looping
    for q_row in range(1, Q + 1):
        for p_row in range(1, P + 1):
            row = (q_row - 1) * P + p_row
            for q_col in range(1, Q + 1):
                for p_col in range(1, P + 1):
                    col = (q_col - 1) * P + p_col
                    p_fft = int(p[p_row - 1] - p[p_col - 1])  # cut - 1 in matlab
                    q_fft = int(q[q_row - 1] - q[q_col - 1])
                    C[row - 1, col - 1] = A[p_0 + p_fft, q_0 + q_fft]
    return C

def complexIdentity(matrixSize):
    """ Wrapper for numpy identity declaration that forces arrays to be complex doubles """
    if matrixSize == 1:
        return 1
    else:
        return np.identity(matrixSize, dtype=np.cdouble);

def redheffer_product(SA: ArrayLike, SB: ArrayLike):
    D = SA[0, 1] @ np.linalg.inv(complexIdentity(SA[0, 0].shape[0]) - SB[0, 0] @ SA[1, 1])
    F = SB[1, 0] @ np.linalg.inv(complexIdentity(SA[0, 0].shape[0]) - SA[1, 1] @ SB[0, 0])
    
    S11 = SA[0, 0] + D @ SB[0, 0] @ SA[1, 0];
    S12 = D @ SB[0, 1];
    S21 = F @ SA[1, 0];
    S22 = SB[1, 1] + F @ SA[1, 1] @ SB[0, 1];
    
    S = np.array([[S11, S12], [S21, S22]])
    return S

def kronecker_delta(i: Union[Number, ArrayLike], j: Union[Number, ArrayLike]) -> Union[Number, ArrayLike]:
    if isinstance(i, Number) and isinstance(j, Number):
        return i == j
    return np.array(i == j).astype(int)

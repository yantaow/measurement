import os
#----use only one core----
nc = "1"
os.environ["OMP_NUM_THREADS"] = nc
os.environ["OPENBLAS_NUM_THREADS"] = nc
os.environ["MKL_NUM_THREADS"] = nc
os.environ["VECLIB_MAXIMUM_THREADS"] = nc
os.environ["NUMEXPR_NUM_THREADS"] = nc

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator, gmres
from infinite_column_mps import iColumnMPS
from tenpy.models.spins import SpinChain
import time
from misc import *
from vumps import *

import pickle
import random
import sys

seed    = int(sys.argv[1])
J       = float(sys.argv[2])
g       = float(sys.argv[3])
D       = int(sys.argv[4])
u       = int(sys.argv[5])
max_iter= float(sys.argv[6])
eps     = float(sys.argv[7])

if seed < 0:
    seed = random.randrange(100)
print("seed:", seed)
np.random.seed(seed)
#----------------------------------------------------------------------------------
if __name__ == "__main__":
#      W = Ising_MPO(J, g, 0)
#      d = 2
#      A0 = normalized_random([d,D,D])
#      As = [A0] * u
#      Ws = [W] * u
#      ALs, ARs, ACs, Cs = normalize(As, verbose=1)
#      ALs, ARs, ACs, Cs = vumps(Ws, ALs, ARs, ACs, Cs, max_iter=max_iter, eps=eps, tol=eps*0.1, verbose=-0)
#      measure(ACs, [[1,0],[0,-1]], verbose=1)
#      print("Exact energy for gx 0.5:", -1.0635444099809814)
#      print("Exact energy for gx 1.5:", -1.671926221536197)
#      print("Exact energy for gx 1.0:", -1.273239544735164)

    '''
    H = \sum_{\langle i,j\rangle, i < j}
          (\mathtt{Jx} S^x_i S^x_j + \mathtt{Jy} S^y_i S^y_j + \mathtt{Jz} S^z_i S^z_j
        + \mathtt{muJ} i/2 (S^{-}_i S^{+}_j - S^{+}_i S^{-}_j))  \\
        - \sum_i (\mathtt{hx} S^x_i + \mathtt{hy} S^y_i + \mathtt{hz} S^z_i) \\
        + \sum_i (\mathtt{D} (S^z_i)^2 + \mathtt{E} ((S^x_i)^2 - (S^y_i)^2))
    '''
    Delta = 3.0
    model_params = {
            'bc_MPS': 'finite',
            'L': 4,
            'conserve': None,
            'Jx': 1,
            'Jy': 1,
            'Jz': Delta,
            }
    XXZ = SpinChain(model_params=model_params)
    ALs, ARs, ACs, Cs = run_vumps(XXZ, u=u, D=D, eps=eps, verbose=0)
    measure(ACs, [[1,0],[0,-1]], verbose=1)
    get_gap(ALs, st='ALs')
    #check_form(ALs, ARs, ACs, Cs)

    print("Delta:", 3, "XXZ exact ground energy:", -0.8310204065300439)




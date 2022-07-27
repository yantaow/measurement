import os
#----use only one core----
nc = "1"
os.environ["OMP_NUM_THREADS"] = nc
os.environ["OPENBLAS_NUM_THREADS"] = nc
os.environ["MKL_NUM_THREADS"] = nc
os.environ["VECLIB_MAXIMUM_THREADS"] = nc
os.environ["NUMEXPR_NUM_THREADS"] = nc

import numpy as np
import tenpy.linalg.np_conserved as npc
from tenpy.models.spins import SpinChain
from tenpy.models.tf_ising import TFIChain
from tenpy.models.hubbard import BoseHubbardModel, BoseHubbardChain
from tenpy.models.hofstadter import HofstadterBosons

from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
from tenpy.algorithms.truncation import svd_theta, TruncationError
from measurement.vumps.vumps import *
from isotns.networks.column_mps import ColumnMPS
from misc import *
import matplotlib.pyplot as plt
import copy

#--------------------------------------
def measurement_entropy(cmps, mu):
    '''
    As : Ground state MPS tensors, in the right canonical form
    mu : measurement strength
    '''
    L = len(cmps) #unit-cell size
    d = cmps[0].shape[0]

    P1      = np.array([[0,0],[0,1]]) #up-projector
    P0      = np.array([[1,0],[0,0]]) #down-projector

    M = np.zeros([d, d, d])     # measurement operators
    # mu = pi/4 corresponds to a projective measurement
    M[0]    = np.sqrt(0.5)*(np.cos(mu)+np.sin(mu))*P1 + np.sqrt(0.5)*(np.cos(mu)-np.sin(mu))*P0
    M[1]    = np.sqrt(0.5)*(np.cos(mu)-np.sin(mu))*P1 + np.sqrt(0.5)*(np.cos(mu)+np.sin(mu))*P0


    assert cmps[0].shape[1] == 1
    T1_all = np.eye(1)
    T2_all = np.eye(1)

    for i in range(L): # sum over sites
        D_l = cmps[i].shape[1]
        D_r = cmps[i].shape[2]
        T1      = np.zeros([D_l**2, D_r**2]) * 1.j
        T2      = np.zeros([D_l**4, D_r**4]) * 1.j
        for s in range(d): # sum over outcomes
            MB    = mT(M[s], cmps[i], 0, order='mT')
            T  = np.tensordot(MB, MB.conj(), [[0], [0]])
            T, _ = group_legs(T, [[0,2],[1,3]])
            T1 += T
            T2 += np.kron(T, T)
        T1_all = T1_all @ T1
        T2_all = T2_all @ T2

#      print('T1_all:', T1_all)
#      print('T2_all:', T2_all)

    return -1 * np.log2(T2_all[0][0])/L
#--------------------------------------
seed    = int(sys.argv[1])
D     = int(sys.argv[2])
u       = int(sys.argv[3])

if seed < 0:
    seed = random.randrange(100)
print("seed:", seed)
np.random.seed(seed)
#--------------------------------------
if __name__ == '__main__':
    L = 200

    A = np.zeros([2,1,1])
    A[0,0,0] = 1/np.sqrt(2)
    A[1,0,0] = 1/np.sqrt(2)
    Xs = []
    for i in range(L):
        Xs.append(copy.deepcopy(A))
    cmps_X = ColumnMPS(list_tensor=Xs)

    AL = np.load('AL.npy')
    get_gap(AL, st='AL', n=5)
    ALs = []
    for i in range(L):
        ALs.append(copy.deepcopy(AL))
    cmps = ColumnMPS(list_tensor=ALs)

    #AL0 = fill_out(A, 2, 2, in_inds=[0,1], out_inds=[2])
    AL0 = random_isometry([2,1,2], [0,1])
    cmps[0] = AL0

    cmps1 = cmps.copy()
    cmps2 = cmps.copy()

    a = 1/np.sqrt(2)
    b = 1/np.sqrt(2)
    r1 = np.zeros([2,1])
    r1[0,0] = a
    r1[1,0] = b
    cmps1[-1] = mT(r1, cmps1[-1], 2, order='Tm')

    r2 = np.zeros([2,1])
    r2[0,0] = 0
    r2[1,0] = 1
    cmps2[-1] = mT(r2, cmps2[-1], 2, order='Tm')

    print('<1|1>:', cmps1.overlap(cmps1.copy()))
    print('<2|2>:', cmps2.overlap(cmps2.copy()))
    print('<1|2>:', cmps1.overlap(cmps2.copy()))
    print('<X|1>:', cmps_X.overlap(cmps1.copy()))
    print('<X|2>:', cmps_X.overlap(cmps2.copy()))
    cmps1.twopt_rdm(L//2, L//2+1)
    cmps2.twopt_rdm(L//2, L//2+1)
    cmps_X.twopt_rdm(L//2, L//2+1)

    mus = np.linspace(0,np.pi/4,110)
    ws = np.zeros([mus.shape[0], 4]) * 1.j
    Ss = []
    for i in range(mus.shape[0]):
        S = measurement_entropy(cmps1, mus[i])
        assert S.imag < 1e-10
        print('i {0:<2}'.format(i), 'S2:', S)
        Ss.append(S.real)

    colors = ['r','g','b','k']
    plt.plot(mus, Ss, color=colors[0])
    plt.xlabel(r'$\mu$')
    plt.ylabel(r'$\lambda$')
    plt.show()

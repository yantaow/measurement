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

def measurement_entropy(ARs, mu):
    '''
    ARs: Ground state MPS tensors, in the right canonical form
    mu : measurement strength
    '''
    N = len(ARs) #unit-cell size
    d = ARs[0].shape[0]

    P1      = np.array([[0,0],[0,1]]) #up-projector
    P0      = np.array([[1,0],[0,0]]) #down-projector
    # measurement operators
    M = np.zeros([d, d, d])
    # mu = pi/4 corresponds to a projective measurement
    M[0]    = np.sqrt(0.5)*(np.cos(mu)+np.sin(mu))*P1 + np.sqrt(0.5)*(np.cos(mu)-np.sin(mu))*P0
    M[1]    = np.sqrt(0.5)*(np.cos(mu)-np.sin(mu))*P1 + np.sqrt(0.5)*(np.cos(mu)+np.sin(mu))*P0

    # tensors in right canonical form as standard numpy arrays
    Bs      = np.zeros([N, chi, d, chi]) * 1.j  #[sites, vL, p, vR]
    for i in range(N):
        Bs[i] = ARs[i].transpose([1,0,2])
#          canon = np.einsum('iak,jak->ij', Bs[i], Bs[i].conj())
#          assert np.max(np.abs(canon-np.identity(chi))) < 1e-10, 'not in right canonical form'

    # transfer matrices generating \sum_m p[m]
    T1      = np.zeros([N, chi**2, chi**2]) * 1.j
    # transfer matrices generating \sum_m p^2[m]
    T2      = np.zeros([N, chi**4, chi**4]) * 1.j
    for i in range(N): # sum over sites
        for j in range(d): # sum over outcomes
            MB    = np.einsum('ab,ibj->iaj', M[j], Bs[i])
            BMMB  = np.einsum('iaj,kal->ikjl',np.conj(MB),MB)
            flatBMMB = BMMB.reshape([chi**2,chi**2])
            T1[i] += flatBMMB
            T2[i] += np.kron(flatBMMB,flatBMMB)
    # transfer matrices for two-site unit cell
    T1cell = T1[0]
    T2cell = T2[0]
    for i in range(1, N):
        T1cell = T1cell @ T1[i]
        T2cell = T2cell @ T2[i]
    print('|T2.imag|:', np.linalg.norm(T2cell.imag))


    # if the measurements are a channel, leading eigenvalue of T1cell is unity
    w1, v1  = np.linalg.eig(T1cell)
#      print('w1:', sorted(w1, key=abs, reverse=True)[:4])
    assert np.abs(1-w1[np.argmax(np.abs(w1))]) < 1e-10, 'measurements not a channel'

    # get spectrum of T2 (which has lots of symmetries I am not yet using)
    w2, v2 = np.linalg.eig(T2cell)
#      print('T2cell:\n', T2cell)
    # N.B. leading w2 = 1/4 for mu=0
    # return leading 3 eigenvalues of T2
    return sorted(w2, key=abs, reverse=True)[:4]

#--------------------------------------
seed    = int(sys.argv[1])
chi     = int(sys.argv[2])
u       = int(sys.argv[3])

if seed < 0:
    seed = random.randrange(100)
print("seed:", seed)
np.random.seed(seed)
#--------------------------------------
if __name__ == '__main__':


    L = 1000


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



    AL0 = fill_out(A, 2, 2, in_inds=[0,1], out_inds=[2])
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
    r2[1,0] = b
    cmps2[-1] = mT(r2, cmps2[-1], 2, order='Tm')

    print('<1|1>:', cmps1.overlap(cmps1.copy()))
    print('<2|2>:', cmps2.overlap(cmps2.copy()))
    print('<1|2>:', cmps1.overlap(cmps2.copy()))
    print('<X|1>:', cmps_X.overlap(cmps1.copy()))
    print('<X|2>:', cmps_X.overlap(cmps2.copy()))
    cmps1.onept_rdm(L//2)
    cmps2.onept_rdm(L//2)



#      mus = np.linspace(0,np.pi/4,11)
#      ws = np.zeros([mus.shape[0], 4]) * 1.j
#
#      for i in range(mus.shape[0]):
#          ws[i] = measurement_entropy(ALs, mus[i])
#          assert ws[i][0].imag < 1e-10
#          ws[i][0] = ws[i][0].real
#          print('i {0:<2}'.format(i), 'lambda:', ws[i], 'S2:', 1/(1-2)*np.log(ws[i][0]).real)
#          # print('i {0:<2}'.format(i), 'lambda:', ws[i], '2nd-3rd:', abs(ws[i][1]-ws[i][2]))
#      colors = ['r','g','b','k']
#      for i in range(4):
#          plt.plot(mus, abs(ws[:,i]),color=colors[i])
#      plt.xlabel(r'$\mu$')
#      plt.ylabel(r'$\lambda$')
    #plt.show()

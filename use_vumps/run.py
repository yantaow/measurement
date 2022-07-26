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
from misc import *
import matplotlib.pyplot as plt

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
        canon = np.einsum('iak,jak->ij', Bs[i], Bs[i].conj())
        assert np.max(np.abs(canon-np.identity(chi))) < 1e-10, 'not in right canonical form'

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
    '''
    H = \sum_{\langle i,j\rangle, i < j}
          (\mathtt{Jx} S^x_i S^x_j + \mathtt{Jy} S^y_i S^y_j + \mathtt{Jz} S^z_i S^z_j
        + \mathtt{muJ} i/2 (S^{-}_i S^{+}_j - S^{+}_i S^{-}_j))  \\
        - \sum_i (\mathtt{hx} S^x_i + \mathtt{hy} S^y_i + \mathtt{hz} S^z_i) \\
        + \sum_i (\mathtt{D} (S^z_i)^2 + \mathtt{E} ((S^x_i)^2 - (S^y_i)^2))
    '''

    model_params = {
            'bc_MPS': 'finite',
            'L': 4,
            'conserve': None,
            'Jx': -1,
            'Jy': 0,
            'Jz': 0.0,
            'hx': 0.0,
            'hy': 0.0,
            }
    model = SpinChain(model_params=model_params)
#      model_params = {
#              'bc_MPS': 'finite',
#              'L': 4,
#              'conserve': None,
#              'J': 1,
#              'g': 0.5,
#              }
#      model = TFIChain(model_params=model_params)


    np.set_printoptions(linewidth=10000)
    ALs, ARs, ACs, Cs = run_vumps(model, u=u, D=chi, eps=1e-12, verbose=-1)
    rdm = np.tensordot(ACs[0], ACs[0].conj(), [[1,2],[1,2]])
    print('rdm:\n', rdm)
    print('ALs[0]:\n', ALs[0][0,:,:])
    print('ALs[1]:\n', ALs[0][1,:,:])

    measure(ACs, [[0,1],[1,0]], verbose=1)
    measure(ACs, [[0,-1.j],[1.j,0]], verbose=1)
    measure(ACs, [[1,0],[0,-1]], verbose=1)
    get_gap(ALs, st='ALs')


    mus = np.linspace(0,np.pi/4,110)
    ws = np.zeros([mus.shape[0], 4]) * 1.j

    for i in range(mus.shape[0]):
        ws[i] = measurement_entropy(ARs, mus[i])
        assert ws[i][0].imag < 1e-10
        ws[i][0] = ws[i][0].real
        print('i {0:<2}'.format(i), 'lambda:', ws[i], 'S2:', 1/(1-2)*np.log(ws[i][0]).real)
        # print('i {0:<2}'.format(i), 'lambda:', ws[i], '2nd-3rd:', abs(ws[i][1]-ws[i][2]))
    colors = ['r','g','b','k']
    for i in range(4):
        plt.plot(mus, abs(ws[:,i]),color=colors[i])
    plt.xlabel(r'$\mu$')
    plt.ylabel(r'$\lambda$')
    plt.show()

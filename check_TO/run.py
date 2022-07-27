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
seed    = int(sys.argv[1])
chi     = int(sys.argv[2])
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

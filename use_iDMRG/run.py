import numpy as np
import tenpy.linalg.np_conserved as npc
from tenpy.models.spins import SpinChain
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
from tenpy.algorithms.truncation import svd_theta, TruncationError
import matplotlib.pyplot as plt

def run(Jz,chi,mu):
    # Jz: anisotropy
    # chi: bond dimension
    # mu: measurement strength

    d = 2
    L = 2 # infinite DMRG
    print("Jz =" + str(Jz), "chi ="+str(chi), "mu = "+str(mu))
    model_params = dict(L=L, Jx=1., Jy=1., Jz=Jz, bc_MPS='infinite', conserve='Sz')
    dmrg_params = {
        'trunc_params': {
            'chi_max': chi,
            'svd_min': 1e-10,
            'trunc_cut': None
        },
        'update_env': 20,
    #    'start_env': 20,
        'max_E_err': 0.0001,
        'max_S_err': 0.0001,
        'mixer': False,
        'max_sweeps': 10000,
        'norm_tol': 1e-4
    }
    M = SpinChain(model_params)

    # initial state fixes total Sz=0 sector
    psi     = MPS.from_product_state(M.lat.mps_sites(),(["up", "down"] * L)[:L], M.lat.bc_MPS)
    engine  = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
    engine.run()
    psi.canonical_form()

    P1      = np.array([[0,0],[0,1]]) #up-projector
    P0      = np.array([[1,0],[0,0]]) #down-projector
    # measurement operators
    M = np.zeros([2, d, d])
    # mu = pi/4 corresponds to a projective measurement
    M[0]    = np.sqrt(0.5)*(np.cos(mu)+np.sin(mu))*P1 + np.sqrt(0.5)*(np.cos(mu)-np.sin(mu))*P0
    M[1]    = np.sqrt(0.5)*(np.cos(mu)-np.sin(mu))*P1 + np.sqrt(0.5)*(np.cos(mu)+np.sin(mu))*P0

    # tensors in right canonical form as standard numpy arrays
    Bs      = np.zeros([L, chi, d, chi])  #[sites, vL, p, vR]
    for i in range(2):
        Bs[i] = (psi.get_B(i)).to_ndarray()
        canon = np.einsum('iak,jak->ij', Bs[i], Bs[i].conj())
        assert np.max(np.abs(canon-np.identity(chi))) < 1e-10, 'not in right canonical form'

    # transfer matrices generating \sum_m p[m]
    T1      = np.zeros((2, chi**2, chi**2))
    # transfer matrices generating \sum_m p^2[m]
    T2      = np.zeros((2, chi**4, chi**4))
    for i in range(2): # sum over sites
        for j in range(2): # sum over outcomes
            MB    = np.einsum('ab,ibj->iaj', M[j], Bs[i])
            BMMB  = np.einsum('iaj,kal->ikjl',np.conj(MB),MB)
            flatBMMB = BMMB.reshape((chi**2,chi**2))
            T1[i] += flatBMMB
            T2[i] += np.kron(flatBMMB,flatBMMB)
    # transfer matrices for two-site unit cell
    T1cell  = T1[0] @ T1[1]
    T2cell  = T2[0] @ T2[1]

    # if the measurements are a channel, leading eigenvalue of T1cell is unity
    w1, v1  = np.linalg.eig(T1cell)
    assert np.abs(1-w1[np.argmax(np.abs(w1))]) < 1e-10, 'measurements not a channel'

    # get spectrum of T2 (which has lots of symmetries I am not yet using)
    w2, v2 = np.linalg.eig(T2cell)
    # N.B. leading w2 = 1/4 for mu=0
    # return leading 3 eigenvalues of T2
    leadingvals2 = np.sort(np.abs(w2))[-3:]
    return leadingvals2
#--------------------------------------
if __name__ == '__main__':
    Jz      = 3.0 # anistropy
    chi     = 5 # bond dimension
    mus     = np.linspace(0,np.pi/4,11)
    # here computing leading 3 eigenvalues of T2
    leadingvals = np.zeros((mus.shape[0],3))
    for i in range(mus.shape[0]):
        print(i)
        leadingvals[i] = run(Jz,chi,mus[i])
    colors = ['r','g','b']
    for i in range(3):
        plt.plot(mus,leadingvals[:,i],color=colors[i])
    plt.xlabel(r'$\mu$')
    plt.ylabel(r'$\lambda$')
    plt.show()

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
import time
from misc import *

import random
import sys

seed    = int(sys.argv[1])
D       = int(sys.argv[2])
u       = int(sys.argv[3])
eps     = float(sys.argv[4])

if seed < 0:
    seed = random.randrange(100)
print("seed:", seed)
np.random.seed(seed)

mv_count = 0
#----------------------------------------------------------------------------------
def TR(A, r, n0=0, W=None):
    '''
        ---    --A--   0---
          |      |        |
         TR  =   |        R
          |      |        |
        ---    --A--   1---

    or
       1---    --A--   1---
          |      |        |
      0--Tr  = --W--   0--r
          |      |        |
       2---    --A--   2---

    Return T_{n0+1}*T_{n0+2} ...T_{n0-1}*T_{n0}*r

    '''
    if not isinstance(A, list):
        A = [A]
    if W is not None and (not isinstance(W, list)):
        W = [W]

    N = len(A)
    if W is None:
        for n in range(N):
            Ar = np.tensordot(r, A[(n0-n)%N], [[0],[2]])
            r = np.tensordot(Ar, A[(n0-n)%N].conj(), [[0,1],[2,0]])
    else:
        for n in range(N):
            Ar = np.tensordot(r, A[(n0-n)%N], [[1],[2]])
            WAr = np.tensordot(Ar, W[(n0-n)%N], [[0,2],[1,3]])
            r = np.tensordot(WAr, A[(n0-n)%N].conj(), [[0,3],[2,0]])
            r = r.transpose([1,0,2])
    return r
#----------------------------------------------------------------------------------
def LT(A, l, n0=0, W=None):
    '''
        ---0 --A--   ---0
        |      |     |
        L      |   = LT
        |      |     |
        ---1 --A--   ---1
    or
       ---1 --A--   ---1
       |      |     |
       l--0 --W-- = lT--0
       |      |     |
       ---2 --A--   ---2

    Return l*T_n0*T_{n0+1} ...*T_{n0-1}
    '''
    if not isinstance(A, list):
        A = [A]
    if W is not None and (not isinstance(W, list)):
        W = [W]

    N = len(A)
    if W is None:
        for n in range(N):
            lA = np.tensordot(l, A[(n+n0)%N], [[0],[1]])
            l = np.tensordot(lA, A[(n+n0)%N].conj(), [[0,1],[1,0]])
    else:
        for n in range(N):
            lA = np.tensordot(l, A[(n+n0)%N], [[1],[1]])
            lAW = np.tensordot(lA, W[(n+n0)%N], [[0,2],[0,3]])
            l = np.tensordot(lAW, A[(n+n0)%N].conj(), [[0,3],[1,0]])
            l = l.transpose([1,0,2])
    return l
#----------------------------------------------------------------------------------
def Ising_MPO(J, g, lam):
    '''
    H = J \sum_j \sum_{n>0} lambda^{n-1} Z_j Z_{j+n} + g \sum \sum_j X_j
    (C15) of arxiv.org/pdf/1701.07035 (with X and Z exchanged)
    W = W[a,b,s,t]
    W_{ab}^{st} = W_{as,bt}, a b are slow indices
    a,b=0,1,2
    s,t=0,1
    W_{0s,0t} = id
    W_{1s,1t} = lambda*id
    W_{2s,2t} = id
    W_{1s,0t} = J * Z
    W_{2s,1t} = Z
    W_{2s,0t} = g * X
    W_others  = 0
    '''
    X = np.array([[0,1],[1,0]])
    Z = np.array([[1,0],[0,-1]])
    I = np.array([[1,0],[0,1]])
    W = np.zeros([3,3,2,2])
    #W[a,b,s,t]
    W[0, 0, :, :] = I
    W[1, 1, :, :] = lam * I
    W[2, 2, :, :] = I
    W[1, 0, :, :] = J*Z
    W[2, 1, :, :] = Z
    W[2, 0, :, :] = g*X

    print("W:\n", to_mat(W, [0,2], [1,3]))
    return W
#----------------------------------------------------------------------------------
def get_lw(Ws, ALs, R, n=0, tol=1e-10, verbose=-1):
    '''
    Algorithm 6 in PRB 97, 045145 (2018)
    Return the left leading generalized eigenvector of TW_AL

    Input:
    1)    t
          |
       a--W--b
          |
          s

       Index structrue: W[a,b,s,t] = [left, right, p, p*]
       t contracts with the MPS physical states, so t is the "bra", and
       s is the "ket".

       In matrix notation,  W_{ab}^{st} = W_{as,bt}, a and b are slow indices.
       For example,
       W =  --- --- ---
           | 1 | 0 | 0 |
            --- --- ---
           | A | B | 0 |
            --- --- ---
           | C | D | 1 |
            --- --- ---
    2) AL: left normal form of an iMPS
        --AL--
          |
          t
    3) R: right eigevector of T_AL

        ---    --AL--   (0)---
          |      |           |
          R  =   |           R
          |      |           |
        ---    --AL--   (1)---

    Output:
    1) lw such that
      ---(1)  ---AL---    ---     ---
      |          |        |       |
      |          |        |       |
     lw--(0)  ---W---  = lw-- + e*I_dW
      |          |        |       |
      |          |        |       |
      ---(2)  ---AL---    ---     ---
    where e is the energy per site.
    I_dW = I if the middle index is dW-1; I_dW = 0, otherwise
    '''
    global mv_count
    N = len(Ws)
    for nn in range(N):
        assert Ws[nn].shape[0] == Ws[nn].shape[1]
        assert Ws[nn].shape[2] == Ws[nn].shape[3]
        assert ALs[nn-1].shape[2] == ALs[nn].shape[1]

    dW = Ws[0].shape[0]
    D  = ALs[0].shape[1]
    d  = ALs[0].shape[0]
    I  = np.eye(D)
    lw = np.zeros([dW, D, D]) * 1.j
    for a in range(dW-1, -1, -1):
        if a == dW-1:
            lw[a,:,:] = I
        else:
            Ya = np.zeros([D,D]) * 1.j
            for b in range(dW-1,a,-1):
                lw_b = lw[b,:,:].reshape([1, D, D])
                Ws_mod = [W.copy() for W in Ws]
                Ws_mod[(n+1)%N] = Ws_mod[(n+1)%N][b,:,:,:].reshape([1,dW,d,d])
                Ws_mod[n] = Ws_mod[n][:,a,:,:].reshape([-1,1,d,d])
                lw_b = LT(ALs, lw_b, n0=n+1, W=Ws_mod)

                Ya += lw_b.reshape([D,D])
            if np.linalg.norm(Ws[0][a,a,:,:]) < 1e-10:
                lw[a,:,:] = Ya
            elif np.linalg.norm(Ws[0][a,a,:,:]-np.eye(d)) < 1e-10:
                #solve lw[a] : (lw[a]| [1-TL+|R)(1|] = Ya - (Ya|R)(1|
                YaR = np.tensordot(Ya, R, [[0,1],[0,1]])
                print("YaR:", YaR.real)
                rhs = Ya - YaR * I
                rhs = rhs.reshape([D**2])
                def mv(l):
                    global mv_count
                    mv_count += 1
                    l = l.reshape([D,D])
                    lT = LT(ALs, l, n0=n+1, W=None)
                    lR = np.tensordot(l, R, [[0,1],[0,1]])
                    return (l - lT + lR*I).reshape([D**2])
                LinOp = LinearOperator((D**2, D**2), matvec=mv)
                mv_0 = mv_count
                lw[a,:,:] = (gmres(LinOp, rhs, maxiter=100, tol=tol)[0]).reshape([D,D])
                mv_1 = mv_count
                if verbose >= 1:
                    print("lw gmres a:", a, ", mv count:", mv_1 - mv_0)
            else:
                print("lambda !=0 case not implemeneted")
                raise NotImplementedError

    lwTw = LT(ALs, lw, n0=n+1, W=Ws)
    if verbose >= 2:
        print("lwTw-lw:\n", (lwTw-lw).real)
    return lw, (lwTw-lw).real[0,0,0]
#----------------------------------------------------------------------------------
def get_rw(Ws, ARs, L, n=0, tol=1e-10, verbose=-1):
    '''
    Algorithm 6 in PRB 97, 045145 (2018)
    Return the left leading generalized eigenvector of TW_AL
    See comments in get_lw()
    '''
    global mv_count
    N = len(Ws)
    for nn in range(N):
        assert Ws[nn].shape[0] == Ws[nn].shape[1]
        assert Ws[nn].shape[2] == Ws[nn].shape[3]
        assert ARs[nn-1].shape[2] == ARs[nn].shape[1]

    dW = Ws[0].shape[0]
    D  = ARs[0].shape[1]
    d  = ARs[0].shape[0]
    I  = np.eye(D)
    rw = np.zeros([dW, D, D]) * 1.j
    for a in range(dW):
        if a == 0:
            rw[a,:,:] = I
        else:
            Ya = np.zeros([D,D]) * 1.j
            for b in range(a):
                rw_b = rw[b,:,:].reshape([1,D,D])
                Ws_mod = [W.copy() for W in Ws]
                Ws_mod[(n+1)%N] = Ws_mod[(n+1)%N][a,:,:,:].reshape([1,dW,d,d])
                Ws_mod[n] = Ws_mod[n][:,b,:,:].reshape([-1,1,d,d])
                rw_b = TR(ARs, rw_b, n0=n, W=Ws_mod)

                Ya += rw_b.reshape([D,D])
            if np.linalg.norm(Ws[0][a,a,:,:]) < 1e-10:
                rw[a,:,:] = Ya
            elif np.linalg.norm(Ws[0][a,a,:,:]-np.eye(d)) < 1e-10:
                #solve rw: [1-T_R+|1)(L|]|r) = Ya-|1)(L|Ya)
                LYa = np.tensordot(L, Ya, [[0,1],[0,1]])
                print("LYa:", LYa.real)
                rhs = Ya - LYa * I
                rhs = rhs.reshape([D**2])
                def mv(r):
                    global mv_count
                    mv_count += 1
                    r = r.reshape([D,D])
                    Tr = TR(ARs, r, n0=n, W=None)
                    Lr = np.tensordot(L, r, [[0,1],[0,1]])
                    return (r - Tr + Lr*I).reshape([D**2])
                LinOp = LinearOperator((D**2, D**2), matvec=mv)
                mv_0 = mv_count
                rw[a,:,:] = (gmres(LinOp, rhs, maxiter=100, tol=tol)[0]).reshape([D,D])
                mv_1 = mv_count
                if verbose >= 1:
                    print("rw gmres mv count:", mv_1 - mv_0)
            else:
                print("lambda !=0 case not implemeneted")
                raise NotImplementedError

    Twrw  = TR(ARs, rw, n0=n, W=Ws)
    if verbose >= 2:
        print("Twrw-rw:\n", (Twrw-rw).real)
    return rw, (Twrw-rw).real[dW-1,0,0]
#----------------------------------------------------------------------------------
def normalize(As, verbose=-1):
    icmps = iColumnMPS(tensors=As)
    info = icmps.to_form(form='A', init='QR', epsilon=1e-15, verbose=-1)
    ALs = icmps.Psi

    info = icmps.to_form(form='B', init='QR', epsilon=1e-15, verbose=-1)
    ARs = icmps.Psi
    Cs = info['Cs']
    ACs = info['ACs']

    if verbose >= 2:
        for n in range(len(ALs)):
            print("-"*10, "check form in normalize", "-"*10)
            print("n:", n)
            #--AC--=--AL--C--
            #  |      |
            AL_C = np.tensordot(ALs[n], Cs[n], [[2],[0]])
            C_AR = np.tensordot(Cs[n-1], ARs[n], [[1],[1]]).transpose([1,0,2])
            AC   = ACs[n]
            print("norm(AC-AL_C):", np.linalg.norm(AC-AL_C))
            print("norm(C_AR-AL_C):", np.linalg.norm(C_AR-AL_C))
            print("norm(C_AR-AC):", np.linalg.norm(C_AR-AC))
            print("check_oc AL:", check_oc(ALs[n], [0,1]))
            print("check_oc AR:", check_oc(ARs[n], [0,2]))
            print("AC norm:", np.linalg.norm(ACs[n]))
            print("C norm:", np.linalg.norm(Cs[n]))
    return ALs, ARs, ACs, Cs
#----------------------------------------------------------------------------------
def apply_H_AC(W, l, r, AC, verbose=-1):
    '''
    Eq. (C28) in PRB 97, 045145 (2018)
    '''
    lA = np.tensordot(l, AC, [[1],[1]])
    lAW = np.tensordot(lA, W, [[0,2],[0,3]])
    AC = np.tensordot(lAW, r, [[1,2],[1,0]])
    return AC.transpose([1,0,2])
#----------------------------------------------------------------------------------
def Lanczos_AC(W, l, r, verbose=-1):
    global mv_count
    d = W.shape[2]
    D = l.shape[1]
    def mv(AC):
        global mv_count
        mv_count += 1
        AC = AC.reshape([d, D, D])
        AC = apply_H_AC(W, l, r, AC)
        return AC.reshape([d*D*D])
    LinOp = LinearOperator((d*D**2, d*D**2), matvec=mv)
    mv_0 = mv_count
    w, v = eigsh(LinOp, k=1, which='SA', return_eigenvectors=True)
    mv_1 = mv_count

    if verbose >= 1:
        print("lanczos AC mv count:", mv_1 - mv_0)

    return v[:, 0].reshape([d,D,D]), w[0]
#----------------------------------------------------------------------------------
def apply_H_C(l, r, C, verbose=-1):
    '''
    Eq. (C29) in PRB 97, 045145 (2018)
    '''
    lC = np.tensordot(l, C, [[1],[0]])
    C = np.tensordot(lC, r, [[0,2],[0,1]])
    return C
#----------------------------------------------------------------------------------
def Lanczos_C(l, r, verbose=-1):
    global mv_count
    D = l.shape[1]
    def mv(C):
        global mv_count
        mv_count += 1
        C = C.reshape([D, D])
        C = apply_H_C(l, r, C)
        return C.reshape([D*D])
    LinOp = LinearOperator((D**2, D**2), matvec=mv)

    mv_0 = mv_count
    w, v = eigsh(LinOp, k=1, which='SA', return_eigenvectors=True)
    mv_1 = mv_count
    if verbose >= 1:
        print("lanczos C mv count:", mv_1 - mv_0)


    return v[:, 0].reshape([D,D]), w[0]
#----------------------------------------------------------------------------------
def split_AC(AC, C, verbose=-1):
    '''
    Eq. (C29) in PRB 97, 045145 (2018)
    '''
    E_AL = np.tensordot(AC.conj(), C, [[2],[1]])
    AL = polar_max_tensor(E_AL, in_inds=[0,1], out_inds=[2])
    eps_L = 2*(1-np.tensordot(E_AL, AL, [[0,1,2],[0,1,2]]).real)

    E_AR = np.tensordot(AC.conj(), C, [[1],[0]]).transpose([0,2,1])
    AR = polar_max_tensor(E_AR, in_inds=[0,2], out_inds=[1])
    eps_R = 2*(1-np.tensordot(E_AR, AR, [[0,1,2],[0,1,2]]).real)

    if verbose >= 2:
        print("split_AC eps_L:", eps_L)
        print("split_AC eps_R:", eps_R)

    return AL, AR, abs(eps_L), abs(eps_R)
#----------------------------------------------------------------------------------
def vumps(Ws, ALs, ARs, Cs, max_iter=100, eps=1e-15, tol=1e-15, verbose=-1):
    '''
    Sequential vumps algorithm in PRB 97, 045145 (2018)

    If |Phi> = Projection of H|Psi> onto the tangent space at |Psi>,
    Then |Phi> = sum_i ...-AL-AL-B_i-AR-AR-...
    --B-- = --H_AC(AC)-- - --AL--H_C(C)--
      |       |              |
    or
    B = AC' - AL*C'
    [|Phi>] = 0 -> |Phi> \propto |Psi> -> B \propto AC -> AC' = AL*C'
    The VUMPS algorithms looks for AL such that AC' = AL*C'
    '''

    j           = 0
    error       = 1
    t_lr        = 0
    t_lanc      = 0
    t_split     = 0
    N = len(ALs)
    while error > eps and j < max_iter:
        if verbose >= 1:
            print("-"*20, "j:", j, "-"*20)
        errors = []
        for n in range(N):
            print("-"*10, "n:", n, "-"*10)
            #need to compute Lw(n-1) and Rw(n)
            L = Cs[n].T.conj()@Cs[n]
            R = Cs[n-1]@Cs[n-1].T.conj()
            t0 = time.time()
            #nn=n-1
            lw_nn, e_L = get_lw(Ws, ALs, R, n=(n-1)%N, tol=tol, verbose=verbose)
            rw_n,  e_R = get_rw(Ws, ARs, L, n=n,       tol=tol, verbose=verbose)
            lw_n    = LT(ALs[n], lw_nn, n0=0, W=Ws[n])
            rw_nn   = TR(ARs[n], rw_n,  n0=0, W=Ws[n])
            t1 = time.time()

            AC_n, lamAC_n = Lanczos_AC(Ws[n], lw_nn, rw_n, verbose=verbose)
            C_n,  lamC_n  = Lanczos_C(lw_n, rw_n, verbose=verbose)
            C_nn, lamC_nn = Lanczos_C(lw_nn, rw_nn, verbose=verbose)

            t2 = time.time()
            ALs[n], _, eps_L, _ = split_AC(AC_n, C_n, verbose=verbose)
            _, ARs[n], _, eps_R = split_AC(AC_n, C_nn, verbose=verbose)
            ACs[n] = AC_n
            Cs[n] = C_n
            Cs[(n-1)%N] = C_nn

            t3 = time.time()
            errors.append(max(eps_L, eps_R))
            t_lr    += t1-t0
            t_lanc  += t2-t1
            t_split += t3-t2
            if verbose >= 1:
                print("error  :", max(eps_L, eps_R))
                print("e_L    :", e_L)
                print("e_R    :", e_R)
                print("e_L/N  :", e_L/N)
                print("e_R/N  :", e_R/N)
                print("t_lr   :", t1-t0)
                print("t_lanc :", t2-t1)
                print("t_split:", t3-t2)
        error = max(errors)
        if verbose >= 0:
            print("-"*40)
            print("error  :", error)
            print("e_L    :", e_L)
            print("e_R    :", e_R)
            print("e_L/N  :", e_L/N)
            print("e_R/N  :", e_R/N)
            print("lamACn :", lamAC_n)
            print("lamCn  :", lamC_n)
            print("lamCn-1:", lamC_nn)
            print("t_lr   :", t1-t0)
            print("t_lanc :", t2-t1)
            print("t_split:", t3-t2)
            print("AC_n.shape:", AC_n.shape)
        j += 1


    if verbose >= -1:
        print("-"*40)
        print("vumps iter   :", j)
        print("e            :", e_L)
        print("e/N          :", e_L/N)
        print("error        :", error)
        print("t lw and rw  :", t_lr)
        print("t lanc       :", t_lanc)
        print("t split      :", t_split)

    return ALs, ARs, ACs, Cs
#----------------------------------------------------------------------------------
def measure(ACs, O, verbose=-1):
    O_avg = []
    N = len(ACs)
    for n in range(N):
        OAC = np.tensordot(O, ACs[n], [[1],[0]])
        AOA = np.tensordot(OAC, ACs[n].conj(), [[0,1,2],[0,1,2]]).real
        O_avg.append(AOA.item())
    if verbose >= 1:
        print("O_avg:\n", O_avg)
    return O_avg
#----------------------------------------------------------------------------------
if __name__ == "__main__":
    d = 2
    J = -1
    g = 1.5
    W = Ising_MPO(J, g, 0.0)
    A0 = normalized_random([d,D,D])
    A1 = normalized_random([d,D,D])
    A2 = normalized_random([d,D,D])

#      As = [A0, A1]
#      Ws = [W, W]
    As = [A0]
    Ws = [W]
    ALs, ARs, ACs, Cs = normalize(As, verbose=2)



    ALs, ARs, ACs, Cs = vumps(Ws, ALs, ARs, Cs, eps=eps, verbose=0)
    print("Exact energy for gx 0.5:", -1.0635444099809814)
    print("Exact energy for gx 1.5:", -1.671926221536197)
    print("Exact energy for gx 1.0:", -1.273239544735164)

    print("ALs[0].shape:", ALs[0].shape)
    Sz = measure(ACs, [[1,0],[0,-1]])
    print("Sz:", Sz)



#      AW = np.tensordot(ALs[0], Ws[0], [[0],[3]])
#      AWA = np.tensordot(AW, ALs[0].conj(), [[4],[0]])
#      Ew = group_legs(AWA, [[3,1,5],[2,0,4]])[0]
#      print("Ew.shape:", Ew.shape)
#      w, v = np.linalg.eig(Ew)
#      sort_index = np.argsort(np.absolute(w))[::-1]
#      v = ((v.T)[sort_index]).T
#      w = w[sort_index]
#      print("w:\n", w)
#      print("v[8]:\n", v[:,8].reshape([3,D,D]))
#      print("v[9]:\n", v[:,9].reshape([3,D,D]))



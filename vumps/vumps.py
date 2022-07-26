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

import pickle
import random
import sys


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
    H = J \sum_j \sum_n lambda^{n-1} Z_j Z_{j+n} + g \sum \sum_j X_j
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
    W[0, 1, :, :] = J*Z
    W[1, 2, :, :] = Z
    W[0, 2, :, :] = g*X

    print("W:\n", to_mat(W, [0,2], [1,3]))
    return W
#----------------------------------------------------------------------------------
def get_lw(Ws, ALs, R, n=0, tol=1e-10, verbose=-1):
    '''
    Algorithm 6 in PRB 97, 045145 (2018)
    Return the left leading generalized eigenvector of TW_AL
    Use TENPY MPO INDEXING!!!!
    Input:
    1)    t
          |
       a--W--b
          |
          s

       Index structrue: W[a,b,s,t]
       t contracts with the MPS physical states, so t is the "bra", and
       s is the "ket".

       In matrix notation,  W_{ab}^{st} = W_{as,bt}, a and b are slow indices.
       For example,
       W =  --- --- ---
           | 1 | A | B |
            --- --- ---
           | 0 | X | D |
            --- --- ---
           | 0 | 0 | 1 |
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
     lw--(0)  ---W---  = lw-- + e*I_0
      |          |        |       |
      |          |        |       |
      ---(2)  ---AL---    ---     ---
    where e is the energy per site.
    I_0 = I if the middle index is 0; = 0, otherwise
    '''
    global mv_count
    N = len(Ws)
    for nn in range(N):
        assert Ws[nn].shape[2] == Ws[nn].shape[3]
        assert Ws[nn].shape[0] == Ws[nn-1].shape[1]
        assert ALs[nn-1].shape[2] == ALs[nn].shape[1]

    dW = Ws[(n+1)%N].shape[0]
    D  = ALs[(n+1)%N].shape[1]
    I  = np.eye(D)
    lw = np.zeros([dW, D, D]) * 1.j

    lw[0,:,:] = I

    #solve lw[x] : (lw[x]|[1-TL^[x]] = Ya[x]
    #x = indices 1,2,3...,dW-2
    #TL[x] = --W[:1,:]-W-W-..-W[:,1:-1]--
    Ws_mod          = [W.copy() for W in Ws]
    Ws_mod[(n+1)%N] = Ws_mod[(n+1)%N][:1,:,:,:]
    Ws_mod[n]       = Ws_mod[n][:,1:-1,:,:]
    Ya              = LT(ALs, lw[:1,:,:], n0=n+1, W=Ws_mod)

    rhs             = Ya.reshape([D**2*(dW-2)])
    Ws_mod[(n+1)%N] = Ws[(n+1)%N][1:-1,:,:,:].copy()
    if N == 1:
        Ws_mod[n]   = Ws_mod[n][:,1:-1,:,:].copy()
    #if N > 1, Ws_mod[n] is already modified

    def mv(l):
        global mv_count
        mv_count += 1
        l = l.reshape([dW-2, D,D])
        lT = LT(ALs, l, n0=n+1, W=Ws_mod)
        return (l - lT).reshape([D**2*(dW-2)])
    LinOp = LinearOperator((D**2*(dW-2), D**2*(dW-2)), matvec=mv)
    mv_0 = mv_count
    lw[1:-1,:,:] = (gmres(LinOp, rhs, tol=tol)[0]).reshape([dW-2, D,D])
    mv_1 = mv_count
    if verbose >= 1:
        print("lw[x] gmres mv count:", mv_1 - mv_0)

    #solve lw[dW-1] : (lw[dW-1]| [1-TL+|R)(1|] = Ya - (Ya|R)(1|
    Ws_mod[(n+1)%N] = Ws[(n+1)%N][:-1,:,:,:].copy()
    if N > 1:
        Ws_mod[n]   = Ws[n][:,-1:,:,:].copy()
    elif N == 1:
        Ws_mod[n]   = Ws_mod[n][:,-1:,:,:]
    Ya              = LT(ALs, lw[:-1,:,:], n0=n+1, W=Ws_mod).reshape([D, D])
    YaR             = np.tensordot(Ya, R, [[0,1],[0,1]])
    if verbose >= 1:
        print("YaR:", YaR.real)
    rhs             = Ya - YaR * I
    rhs             = rhs.reshape([D**2])
    def mv(l):
        global mv_count
        mv_count += 1
        l = l.reshape([D,D])
        lT = LT(ALs, l, n0=n+1, W=None)
        lR = np.tensordot(l, R, [[0,1],[0,1]])
        return (l - lT + lR*I).reshape([D**2])
    LinOp = LinearOperator((D**2, D**2), matvec=mv)
    mv_0 = mv_count
    lw[-1,:,:] = (gmres(LinOp, rhs, maxiter=10, tol=tol)[0]).reshape([D,D])
    mv_1 = mv_count
    if verbose >= 1:
        print("lw[0] gmres mv count:", mv_1 - mv_0)

    lwTw = LT(ALs, lw, n0=n+1, W=Ws)
    if verbose >= 2:
        diff = lwTw-lw
        mask = abs(diff) > 1e-10
        print("lwTw-lw nonzero:", diff[mask].real)
    return lw, (lwTw-lw).real[-1,0,0]
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
        assert Ws[nn].shape[2] == Ws[nn].shape[3]
        assert Ws[nn].shape[0] == Ws[nn-1].shape[1]
        assert ARs[nn-1].shape[2] == ARs[nn].shape[1]

    dW = Ws[n].shape[1]
    D  = ARs[n].shape[2]
    I  = np.eye(D)
    rw = np.zeros([dW, D, D]) * 1.j

    rw[-1,:,:] = I

    #solve rw[x]: [1-TR^[x]]|rw[x])= |Ya[x])
    Ws_mod          = [W.copy() for W in Ws]
    Ws_mod[n]       = Ws_mod[n][:,-1:,:,:]
    Ws_mod[(n+1)%N] = Ws_mod[(n+1)%N][1:-1,:,:,:]
    Ya              = TR(ARs, rw[-1:,:,:], n0=n, W=Ws_mod)
    rhs             = Ya.reshape([D**2*(dW-2)])
    Ws_mod[n]       = Ws[n][:,1:-1,:,:].copy()
    if N == 1:
        Ws_mod[(n+1)%N]   = Ws_mod[n][1:-1,:,:,:].copy()

    def mv(r):
        global mv_count
        mv_count += 1
        r = r.reshape([dW-2,D,D])
        Tr = TR(ARs, r, n0=n, W=Ws_mod)
        return (r - Tr).reshape([D**2*(dW-2)])
    LinOp = LinearOperator((D**2*(dW-2), D**2*(dW-2)), matvec=mv)
    mv_0 = mv_count
    rw[1:-1,:,:] = (gmres(LinOp, rhs, tol=tol)[0]).reshape([dW-2, D,D])
    mv_1 = mv_count
    if verbose >= 1:
        print("rw[x] gmres mv count:", mv_1 - mv_0)

    #solve rw[0]: [1-T_R+|1)(L|]|r) = Ya-|1)(L|Ya)
    Ws_mod[n]       = Ws[n][:,1:,:,:].copy()
    if N > 1:
        Ws_mod[(n+1)%N] = Ws[(n+1)%N][:1,:,:,:]
    elif N == 1:
        Ws_mod[(n+1)%N] = Ws_mod[(n+1)%N][:1,:,:,:]

    Ya              = TR(ARs, rw[1:,:,:], n0=n, W=Ws_mod).reshape([D, D])
    LYa             = np.tensordot(L, Ya, [[0,1],[0,1]])
    if verbose >= 1:
        print("LYa:", LYa.real)
    rhs             = Ya - LYa * I
    rhs             = rhs.reshape([D**2])
    def mv(r):
        global mv_count
        mv_count += 1
        r = r.reshape([D,D])
        Tr = TR(ARs, r, n0=n, W=None)
        Lr = np.tensordot(L, r, [[0,1],[0,1]])
        return (r - Tr + Lr*I).reshape([D**2])
    LinOp = LinearOperator((D**2, D**2), matvec=mv)
    mv_0 = mv_count
    rw[0,:,:] = (gmres(LinOp, rhs, maxiter=10, tol=tol)[0]).reshape([D,D])
    mv_1 = mv_count
    if verbose >= 1:
        print("rw[0] gmres mv count:", mv_1 - mv_0)

    Twrw  = TR(ARs, rw, n0=n, W=Ws)
    if verbose >= 2:
        diff = Twrw-rw
        mask = abs(diff) > 1e-10
        print("Twrw-rw nonzero:", diff[mask].real)
    return rw, (Twrw-rw).real[0,0,0]
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
        check_form(ALs, ARs, ACs, Cs)
    return ALs, ARs, ACs, Cs
#----------------------------------------------------------------------------------
def check_form(ALs, ARs, ACs, Cs):
    N = len(ALs)
    for n in range(len(ALs)):
        print("-"*10, "check form in normalize", "-"*10)
        print("n:", n)
        #--AC--=--AL--C--
        #  |      |
        AL_C = np.tensordot(ALs[n], Cs[n], [[2],[0]])
        C_AR = np.tensordot(Cs[(n-1)%N], ARs[n], [[1],[1]]).transpose([1,0,2])
        AC   = ACs[n]
        print("norm(AC-AL_C):", np.linalg.norm(AC-AL_C))
        print("norm(C_AR-AL_C):", np.linalg.norm(C_AR-AL_C))
        print("norm(C_AR-AC):", np.linalg.norm(C_AR-AC))
        print("check_oc AL:", check_oc(ALs[n], [0,1]))
        print("check_oc AR:", check_oc(ARs[n], [0,2]))
        print("AC norm:", np.linalg.norm(ACs[n]))
        print("C norm:", np.linalg.norm(Cs[n]))
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
    try:
        mv_0 = mv_count
        w, v = eigsh(LinOp, k=1, which='SA', return_eigenvectors=True)
        mv_1 = mv_count
        if verbose >= 1:
            print("lanczos AC mv count:", mv_1 - mv_0)

    except:
        lW = np.tensordot(W, l, [[0],[0]])
        lWr = np.tensordot(lW, r, [[0], [0]])
        H_AC,_ = group_legs(lWr, [[0,3,5],[1,2,4]])
#          print("H_AC-H_AC*:", np.linalg.norm(H_AC-H_AC.T.conj()))
        w, v = np.linalg.eig(H_AC)
        sort_index = np.argsort(np.absolute(w))
        v = ((v.T)[sort_index]).T
        w = w[sort_index]
        print("w:", w)

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
    try:
        mv_0 = mv_count
        w, v = eigsh(LinOp, k=1, which='SA', return_eigenvectors=True)
        mv_1 = mv_count
        if verbose >= 1:
            print("lanczos C mv count:", mv_1 - mv_0)
    except:
        lr = np.tensordot(l, r, [[0],[0]])
        H_C,_ = group_legs(lr, [[1,3],[0,2]])
        print("H_C-H_C*:", np.linalg.norm(H_C-H_C.T.conj()))
        w, v = np.linalg.eig(H_C)
        sort_index = np.argsort(np.absolute(w))
        v = ((v.T)[sort_index]).T
        w = w[sort_index]
        print("w:", w)

    v[:, 0] = fix_phase(v[:,0])
    return v[:, 0].reshape([D,D]), w[0]
#----------------------------------------------------------------------------------
def split_AC(AC, C, verbose=-1):
    '''
    Eq. (19) in PRB 97, 045145 (2018)
    or
    Eq. (143)-(145) in arxiv.1810.07006
    '''
#      E_AL = np.tensordot(AC.conj(), C, [[2],[1]])
#      AL = polar_max_tensor(E_AL, in_inds=[0,1], out_inds=[2])
#      E_AR = np.tensordot(AC.conj(), C, [[1],[0]]).transpose([0,2,1])
#      AR = polar_max_tensor(E_AR, in_inds=[0,2], out_inds=[1])

    AC_l, pipe_l = group_legs(AC, [[0,1],[2]])
    AC_r, pipe_r = group_legs(AC, [[1],[0,2]])
    U_AC_l, _ = scipy.linalg.polar(AC_l, side='right')
    U_C_l,  _ = scipy.linalg.polar(C, side='right')
    U_C_r,  _ = scipy.linalg.polar(C, side='left')
    U_AC_r, _ = scipy.linalg.polar(AC_r, side='left')
    AL = ungroup_legs(U_AC_l @ U_C_l.conj().T, pipe_l)
    AR = ungroup_legs(U_C_r.conj().T @ U_AC_r, pipe_r)

    eps_L = np.linalg.norm(AC - mT(C, AL, 2, order='Tm'))
    eps_R = np.linalg.norm(AC - mT(C, AR, 1, order='mT'))
    if verbose >= 2:
        print("split_AC eps_L:", eps_L)
        print("split_AC eps_R:", eps_R)

    return AL, AR, abs(eps_L), abs(eps_R)
#----------------------------------------------------------------------------------
def vumps(Ws, ALs, ARs, ACs, Cs, max_iter=100, eps=1e-15, tol=1e-15, verbose=-1):
    '''
    If |Phi> = Projection of H|Psi> onto the tangent space at |Psi>,
    Then |Phi> = sum_i ...-AL-AL-B_i-AR-AR-...
    --B-- = --H_AC(AC)-- - --AL--H_C(C)--
      |       |              |
    or
    B = AC' - AL*C'
    [|Phi>] = 0 -> |Phi> \propto |Psi> -> B \propto AC -> AC' = AL*C'
    The VUMPS algorithms looks for AL such that AC' = AL*C'
    '''
    assert isinstance(Ws, list)
    j           = 0
    error       = 1
    t_lr        = 0
    t_lanc      = 0
    t_split     = 0
    N = len(ALs)
    min_iter    = 2
    while j < min_iter or (error > eps and j < max_iter):
        if verbose >= 0:
            print("-"*20, "j:", j, "-"*20)
        errors = []
        for n in range(N):
            if verbose >= 1:
                print("-"*10, "n:", n, "; j:", j, "-"*10)
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
            print("error  :", error)
            print("eps_L  :", eps_L)
            print("eps_R  :", eps_R)
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
def swap_index(Ws, n, i, j):
    Ws[n][[i,j],:,:,:] = Ws[n][[j,i],:,:,:]
    Ws[n-1][:,[i,j],:,:] = Ws[n-1][:,[j,i],:,:]

    return Ws
#----------------------------------------------------------------------------------
def swap_Ws(Ws, IdL, IdR):
    N = len(Ws)
    for n in range(N):
        Ws[n][[0,IdL[n]],:,:,:] = Ws[n][[IdL[n],0],:,:,:]
        Ws[n-1][:,[0,IdL[n]],:,:] = Ws[n-1][:,[IdL[n],0],:,:]

        Ws[n][[-1,IdR[n-1]],:,:,:] = Ws[n][[IdR[n-1],-1],:,:,:]
        Ws[n-1][:,[-1,IdR[n-1]],:,:] = Ws[n-1][:,[IdR[n-1],-1],:,:]
    return Ws
#----------------------------------------------------------------------------------
def check_Ws(Ws, swap=False):
    print("-"*10, "check Ws", "-"*10)
    if swap:
        Ws_new = [W.copy() for W in Ws]
    N = len(Ws)
    for m in range(N):
        W = Ws[m]
        print("m:", m)
        for n in range(1, N):
            W = np.tensordot(W, Ws[(m+n)%N], [[1],[0]])
            W, _ = group_legs(W, [[0],[3],[1,4],[2,5]])
        print("W.shape:", W.shape)

        d = W.shape[2]
        W_norm = np.zeros([W.shape[0], W.shape[1]])
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                mat = W[i,j,:,:]
                W_norm[i, j] = int(np.linalg.norm(mat) > 1e-10)
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                is_id = np.linalg.norm(W[i,j,:,:]-np.eye(d)) < 1e-8
                if is_id and i == j:
                    print("i, j:", i, j)
                    #print("W_norm[i,:]:", np.linalg.norm(W_norm[i,:])**2)
                    #print("W_norm[:,j]:", np.linalg.norm(W_norm[:,j])**2)
                    if swap and np.isclose(sum(W_norm[:, i]), 1):
                        Ws_new = swap_index(Ws, m, 0, i)
                    elif swap:
                        Ws_new = swap_index(Ws, m, -1, i)
    if swap:
        return Ws_new
#----------------------------------------------------------------------------------
def trace_W(Ws):
    print("-"*10, "trace W", "-"*10)
    N = len(Ws)
    Ws_new = []
    for n in range(N):
        W = Ws[n]
        WW = np.zeros([W.shape[0], W.shape[1]])
        print("n:", n)
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                mat = W[i,j,:,:]
                if np.linalg.norm(mat-np.eye(mat.shape[0])) < 1e-8:
                    WW[i,j] = 1
                elif np.linalg.norm(mat) < 1e-8:
                    WW[i,j] = 0
                else:
                    WW[i,j] = 10
        Ws_new.append(WW.reshape([W.shape[0], W.shape[1], 1, 1]))
    return Ws_new
#----------------------------------------------------------------------------------
def check_UpperTriangular(Ws):
    for W in Ws:
        print(W.shape)
    Ws_tr = trace_W(Ws)
    check_Ws(Ws_tr)
#----------------------------------------------------------------------------------
def check_LWRW(Ws, ALs, ARs, Cs):
    N = len(ALs)
    for n in range(N):
        print("-"*10, "n:", n, "-"*10)
        R = Cs[n]@Cs[n].T.conj()
        lw, e = get_lw(Ws, ALs, R, n=n, tol=1e-10, verbose=2)
        L = Cs[n].T.conj()@Cs[n]
        get_rw(Ws, ARs, L, n=n, tol=1e-10, verbose=2)
        print("e/N:", e/N)
#----------------------------------------------------------------------------------
def get_mpo(tenpy_model, verbose=True):
    """
        Return a list [W_i] of the MPO tensors.
        This is a direct translation of TenPy MPO to pure numpy MPO.

        Return:
            list[np.array]: indices [w_left, w_right, phys, phys*]
    """
    tenpy_mpo = tenpy_model.calc_H_MPO()
    mpo = [W.to_ndarray() for W in tenpy_mpo._W]
    starting_idx = tenpy_mpo.IdL[0]
    ending_idx = tenpy_mpo.IdR[-1]
    mpo[0] = mpo[0][starting_idx : starting_idx + 1, :, :, :]
    mpo[-1] = mpo[-1][:, ending_idx : ending_idx + 1, :, :]
    # To deal with the dangling leg of the mpo[0] and mpo[-1],
    # alternatively, one can set the 'insert_all_id' to False in
    # calc_H_MPO of tenpy.
    if verbose:
        for i in range(len(mpo)):
            print('i:\n', to_mat(mpo[i], [0,2], [1,3]))
    return mpo
#----------------------------------------------------------------------------------
def run_vumps(tenpy_model, u=1, D=10, max_iter=1000, eps=1e-14, verbose=-1):
    W = get_mpo(tenpy_model, verbose=False)[1]
    d = W.shape[-1]
    A0 = normalized_random([d,D,D])
    As = [A0] * u
    Ws = [W] * u
    ALs, ARs, ACs, Cs = normalize(As, verbose=-1)
    ALs, ARs, ACs, Cs = vumps(Ws, ALs, ARs, ACs, Cs, max_iter=max_iter, eps=eps, tol=eps*0.1, verbose=verbose)
    return ALs, ARs, ACs, Cs


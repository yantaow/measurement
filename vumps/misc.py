import pickle
import numpy as np
import scipy as sp
import scipy.linalg
import itertools
import cmath

#----------------------------------------------------------------------------------
def fix_phase(A):
    '''
    fix the phase of the tensor t such that it's largest element (in abs)
    is real and positive
    '''
    reshaped = False
    if A.ndim > 1:
        shape = A.shape
        A = A.reshape([-1])
        reshaped = True
    maxA = A[abs(A).argmax()]
    phase = get_phase(maxA)

    if reshaped:
        A = A.reshape(shape)
    return A/phase
#----------------------------------------------------------------------
def to_mat(W, row_inds, col_inds):
    return group_legs(W, [row_inds, col_inds])[0]
#------------------------------------------------------------------------------
def get_gap(psi, phys_inds=[0,1], verbose=0, st=None, n=2, tol=1e-10, sparse=True):
    '''
    psi is grouped such that
        1--A--2  ---|
           |        |
           0        |
                    S
           0        |
           |        |
        1--A--2  ---|
    '''
    assert isinstance(psi, list)
    if not isinstance(psi, list):
        psi = [psi]
    num_p = psi[0].ndim - 2
    phys_inds = list(range(num_p))
    grouping = [phys_inds, [len(phys_inds)], [len(phys_inds)+1]]
    psi = [group_legs(t, grouping)[0] for t in psi]

    N = len(psi)
    D = psi[0].shape[1]
    try:
        def mv(l):
            l = l.reshape([D,D])
            for i in range(N):
                l_psi = np.tensordot(l, psi[i], [[0],[1]])
                l = np.tensordot(l_psi, psi[i].conj(), [[0,1],[1,0]])
            return l.reshape([D**2])
        LinOp = sp.sparse.linalg.LinearOperator((D**2, D**2), matvec=mv)
        w = sp.sparse.linalg.eigs(LinOp, k=n, which='LM', tol=tol, return_eigenvectors=False)
    except:
        print("exception in get_gap")
        return -1

    eig_T = sorted(w, key=abs, reverse=True)
    gap = abs(eig_T[0]) - abs(eig_T[1])
    if st is not None:
        print(st+" spectrum  :", np.absolute(eig_T)[0:n])
    return gap
#------------------------------------------------------------------------------
def get_fidelity(A, B, A_pinds=[0,1], B_pinds=[0,1], tol=1e-5):
    '''
    A, B is grouped such that
        1--A--2  ---|
           |        |
           0        |
                    S
           0        |
           |        |
        1--B--2  ---|
    '''
    grouping = [A_pinds, [len(A_pinds)], [len(A_pinds)+1]]
    A = group_legs(A, grouping)[0]

    grouping = [B_pinds, [len(B_pinds)], [len(B_pinds)+1]]
    B = group_legs(B, grouping)[0]
    assert A.shape[1] == A.shape[2] and B.shape[1] == B.shape[2]

#      print("A.shape:", A.shape)
#      print("B.shape:", B.shape)

    n = 2
    try:
        def T(S):
            AS = np.tensordot(A, S, [[2], [0]])
            ASB = np.tensordot(AS, B.conj(), [[0,2], [0,2]])
            return ASB
        def mv(S):
            S = S.reshape(A.shape[2], B.shape[2])
            return T(S)
        DA = A.shape[1]
        DB = B.shape[1]
        LinearOp = sp.sparse.linalg.LinearOperator((DA*DB, DA*DB), matvec=mv)
        w = sp.sparse.linalg.eigs(LinearOp, k=n, which='LM', tol=tol, \
                return_eigenvectors=False, maxiter=50)
    except:
        T = np.tensordot(A, B.conj(), [A_pinds, B_pinds])
        T = group_legs(T, [[1,3],[0,2]])[0]
        w = sp.sparse.linalg.eigs(T, k=n, which='LM', tol=tol, \
                return_eigenvectors=False, maxiter=50)
    eig_T = sorted(w, key=abs, reverse=True)
    return  abs(eig_T[0])**2
#------------------------------------------------------------
def polar_max(E):
    '''argmax_{a isometry} Tr(a E) = (UV)^dag, if E = UsV'''
    assert E.shape[0] <= E.shape[1], "E.shape:{}".format(E.shape)
    U,s,V = svd(E, compute_uv=True, full_matrices=False)
#      print("Tr(s_E)^2:", sum(s)**2)
    return (U@V).conj().T
#------------------------------------------------------------
def polar_max_tensor(E, in_inds=[0], out_inds=[1]):
    '''
    Return argmax_{a isometry} tTr(a,E) as a tensor.
    in_inds are incoming indices of a; and out_inds are outgoing
    indices of a.
    '''
    E, pipe = group_legs(E, [in_inds, out_inds])
    a = polar_max(E.T)
    return ungroup_legs(a, pipe)
#------------------------------------------------------------
def fill_in(T, index=0, new_dim=4):
    '''
    Enlarge the leg, T[index], to a new dimension by zero padding.
    '''
    inds = list(range(T.ndim))
    inds.pop(index)
    old_dim = T.shape[index]
    assert old_dim <= new_dim
    T, pipe = group_legs(T, [[index], inds])
    if np.any(np.iscomplex(T)):
        print("T.imag norm:", np.linalg.norm(T.imag))
        T_new = np.zeros((new_dim, T.shape[1])) * 1.j
    else:
        T_new = np.zeros((new_dim, T.shape[1]))
    T_new[:old_dim, :] = T
    return ungroup_legs(T_new, pipe)
#------------------------------------------------------------
def fill_out(T, index, new_dim, in_inds=[0,2], out_inds=[1,3]):
    '''
    Enlarge the outgoing leg, T[index], to a new dimension while keeping isometry.
    Enlarge by adding orthogonal complements.
    The input index is the slowest index. For the other out indices, the earlier
    in out_inds, the slower.
    '''
    extend = False
    if len(out_inds) == 1:
        out_inds.append(T.ndim)
        T = T.reshape(list(T.shape) + [1])
        extend = True
    out_inds_2 = out_inds.copy()
    out_inds_2.remove(index)

    old_dim = T.shape[index]
    T, pipe = group_legs(T, [in_inds, [index], out_inds_2])

    T_new = T.reshape([T.shape[0], T.shape[1]*T.shape[2]])
    T_new = add_col_complement(T_new, new_dim*T.shape[2])
    T_new = T_new.reshape([T_new.shape[0], new_dim, T.shape[2]])
    T_new = ungroup_legs(T_new, pipe)

    if extend:
        orig_shape = T_new.shape[:-1]
        T_new = T_new.reshape(orig_shape)
    return T_new

#------------------------------------------------------------
def add_col_complement(V, new_dim):
    '''
    V is an np array matrix.
    Return a matrix whose column vectors are add_dim number
    of vectors orthogonal to the column vectors in V.
    Use modified Gram-Schmidt.

    Inserts random vectors in the case of linearly dependent rows."""
    '''
    old_dim = V.shape[1]
    assert new_dim >= old_dim
    assert new_dim <= V.shape[0]

    V = V.T

    # V[i] = ith column of the isometry
    for k in range(old_dim, new_dim):
        norm = 0
        tt = 1
        while norm < 1e-10:
            Vk = normalized_random([V.shape[1]])
            for j in range(k):
                Vk = Vk - V[j] * (np.conj(V[j]) @ Vk)
            norm = np.linalg.norm(Vk)
            tt += 1
        V = np.append(V, [Vk/norm], axis=0)
    return V.T
#------------------------------------------------------------
def isofill(A, B, iA, iB, in_inds=[0,2], new_d=None, random=True):
    '''
    A-->--B
    Enlarge the bond dimension to new_d.
    The bond is out for A, and in for B.
    in_inds are the in inds of A.
    Optionally, multiply a random unitary on the bond
    '''
    AB_same = id(A) == id(B)
    assert A.shape[iA] == B.shape[iB]
    assert new_d >= A.shape[iA]
    if new_d is None:
        new_d = A.shape[iA]

    A_old = A.copy()
    B_old = B.copy()

    if new_d > A.shape[iA]:
        out_inds = list(set(list(range(A.ndim))) - set(in_inds))
        if AB_same:
            B = fill_in(B, iB, new_d)
            B = fill_out(B, iA, new_d, in_inds, out_inds)
            A = B
        else:
            B = fill_in(B, iB, new_d)
            A = fill_out(A, iA, new_d, in_inds, out_inds)

    if random:
        U = random_orthogonal(new_d)
        if AB_same:
            B = mT(U, B, iB, order='mT')
            B = mT(U.T.conj(), B, iA, order='Tm')
            A = B
        else:
            B = mT(U, B, iB, order='mT')
            A = mT(U.T.conj(), A, iA, order='Tm')

    if not AB_same:
        AB_old = np.tensordot(A_old, B_old, [[iA], [iB]])
        AB = np.tensordot(A, B, [[iA], [iB]])
#          print("dif:", np.linalg.norm(AB-AB_old))
#          assert np.linalg.norm(AB-AB_old) < 1e-5

    return A, B
#------------------------------------------------------------
def mT(m, T, iT, order='mT'):
    '''
    matrix tensor multiplication
    The tensor's inds order doesn't change.
    '''
    assert m.ndim == 2
    T_inds = list(range(T.ndim))
    T_inds.remove(iT)

    if order == 'mT':
        T, pipe = group_legs(T, [[iT], T_inds])
        mxT = m @ T
        return ungroup_legs(mxT, pipe)
    elif order == 'Tm':
        T, pipe = group_legs(T, [T_inds, [iT]])
        Txm = T @ m
        return ungroup_legs(Txm, pipe)
#------------------------------------------------------------
def random_orthogonal(n):
    '''Haar random nxn orthogonal matrix'''
    H = np.random.randn(n, n)
    Q, _ = unique_qr(H)
    return Q
#------------------------------------------------------------
def normalized_random(shape, dist='uniform'):
    if dist == 'uniform':
        tensor = np.random.random(shape).real
    elif dist == 'gaussian':
        tensor = np.random.normal(size=shape).real

    return tensor/np.linalg.norm(tensor)
#------------------------------------------------------------
def max_array(arr, n):
    arr_flat = np.ndarray.flatten(arr)
    sort_ind = np.argsort(arr_flat)[-1:-n:-1]
    max_arr = arr_flat[sort_ind]
    return sort_ind, max_arr
#------------------------------------------------------------
def to_array(Psi, func=None):
    if func is not None:
        T = [[f(t) for f in func] for t in Psi]
    else:
        T = Psi.copy()
    T = np.asarray(T)
    T = np.flip(T, axis=0)
    return T
#------------------------------------------------------------
def check_oc(t, contract_axes):
    l = t.ndim - len(contract_axes) #number of free indices after contraction
    t = np.tensordot(t.conj(), t, axes=[contract_axes, contract_axes])
    if t.ndim > 0:
        t, pipe = group_legs(t, [list(range(0,l)), list(range(l,2*l))])
    else:
        t = np.asarray([t])
    return np.linalg.norm(t - np.eye(t.shape[0]))
#      return np.linalg.norm(t - np.eye(t.shape[0])) / np.sqrt(t.shape[0])
#------------------------------------------------------------
def tensor_qr(t, contract_axes):
    #Assuem the input tensor t is tall, i.e. the Q will be isometric
    #i.e. dim(contract) >= dim(uncontract)
    #contract_axes needs to be sorted
    #Returns Q, R
    #t = Q * R, where Q.shape = t.shape, and Q is isometry when
    #contracted on contract_axes.
    contract_dims = [t.shape[i] for i in contract_axes]
    free_axes = list(set(list(range(0, t.ndim)))-set(contract_axes))
    free_dims = [t.shape[i] for i in free_axes]
    t = t.transpose(contract_axes + free_axes)
    t = t.reshape(np.prod(contract_dims), -1)
    Q, _, R = svd(t)
#      Q, R = unique_qr(t)
    Q = Q.reshape(contract_dims + free_dims)
    R = R.reshape(free_dims + free_dims)
    Q = Q.transpose(invertseq(contract_axes + free_axes))
    return Q, R
#------------------------------------------------------------
def invertseq(s):
    n = len(s)
    invs = [-1] * n
    for i in range(n):
        invs[s[i]] = i
    return invs
#------------------------------------------------------------
def get_phase(r):
    if abs(r) < 1e-16:
        return 1
    else:
        return r/abs(r)
#------------------------------------------------------------
def qr_trunc(theta, D):
    '''
    Normalization is forced: norm(returned QR) = 1
    '''
    assert abs(np.linalg.norm(theta) - 1) < 1e-10

    #theta.P = Q.R
    Q, R, P = scipy.linalg.qr(theta, mode='economic', pivoting=True)

    phase = [get_phase(r) for r in np.diag(R)]
    phase_inv = [1./p for p in phase]
    phase = np.diag(phase)
    phase_inv = np.diag(phase_inv)

    Q = Q @ phase
    R = phase_inv @ R

    if D < Q.shape[1]:
        Q = Q[:, :D]
        R = R[:D, :]
    R_perm = R[:, invertseq(P)]
    R_perm = R_perm/np.linalg.norm(R_perm)
    p_trunc = scipy.linalg.norm(theta-Q.dot(R_perm))**2

    info = {'p_trunc': p_trunc, 'nrm':1}
    return Q, R_perm, info
#------------------------------------------------------------
def unique_qr(A):
    Q, R = np.linalg.qr(A)
    phase = np.diag([get_phase(r) for r in np.diag(R)])
    phase_inv = np.diag([get_phase(r) for r in np.diag(R)])

    Q = Q @ phase
    R = phase_inv @ R
    assert np.linalg.norm(Q@R- A) < 1e-10
    return Q, R
#------------------------------------------------------------
def make_U(H, t):
    """ U = exp(-t H) """
    #H[0][0] = h1 -- a 4-leg tensor
    d = H[0][0].shape[0]
    return [[
        sp.linalg.expm(-t * h.reshape((d**2, -1))).reshape([d] * 4)
        for h in Hc] for Hc in H]
#------------------------------------------------------------
###Tensor stuff
def rotT(T):
    """ 90-degree counter clockwise rotation of tensor """
    return np.transpose(T, [0, 4, 3, 1, 2])
#------------------------------------------------------------
def group_legs(a, axes):
    """ Given list of lists like axes = [ [l1, l2], [l3], [l4 . . . ]]

		does a transposition of np.array "a" according to l1 l2 l3... followed by a reshape according to parantheses.

		Return the reformed tensor along with a "pipe" which can be used to undo the move
	"""

    nums = [len(k) for k in axes]

    flat = []
    for ax in axes:
        flat.extend(ax)

    a = np.transpose(a, flat)
    perm = np.argsort(flat)

    oldshape = a.shape

    shape = []
    oldshape = []
    m = 0
    for n in nums:
        shape.append(np.prod(a.shape[m:m + n]))
        oldshape.append(a.shape[m:m + n])
        m += n

    a = np.reshape(a, shape)

    pipe = (oldshape, perm)

    return a, pipe
#------------------------------------------------------------
def ungroup_legs(a, pipe):
    """
		Given the output of group_legs,  recovers the original tensor (inverse operation)

		For any singleton grouping [l],  allows the dimension to have changed (the new dim is inferred from 'a').
	"""
    if a.ndim != len(pipe[0]):
        raise ValueError
    shape = []
    for j in range(a.ndim):
        if len(pipe[0][j]) == 1:
            shape.append(a.shape[j])
        else:
            shape.extend(pipe[0][j])

    a = a.reshape(shape)
    a = a.transpose(pipe[1])
    return a
#------------------------------------------------------------
#### MPO STUFF
def transpose_mpo(Psi):
    """Transpose row / column (e.g. bra / ket) of an MPO"""
    return [b.transpose([1, 0, 2, 3]) for b in Psi]
#------------------------------------------------------------
def svd(theta, compute_uv=True, full_matrices=False, fix_gauge=True):
    """SVD with gesvd backup"""
    try:
        U,s,Vh = sp.linalg.svd(theta,
                             compute_uv=compute_uv,
                             full_matrices=full_matrices)
    except np.linalg.linalg.LinAlgError:
        print("*gesvd*")
        U,s,Vh = sp.linalg.svd(theta,
                             compute_uv=compute_uv,
                             full_matrices=full_matrices,
                             lapack_driver='gesvd')
    if fix_gauge:
        maxV = [v[abs(v).argmax()] for v in Vh]
        phase = [1/get_phase(r) for r in maxV]
        phase_inv = [1./p for p in phase]
        phase = np.diag(phase)
        phase_inv = np.diag(phase_inv)
        Vh = phase @ Vh #fixes rows of Vh
        U = U @ phase_inv #changes columns of U accordingly

    assert np.linalg.norm(U@np.diag(s)@Vh-theta) < 1e-10
    return U,s,Vh
#------------------------------------------------------------
def hosvd(Psi, mode='svd'):
    """Higher order SVD. Given rank-l wf Psi, computes Schmidt spectrum Si on each leg-i, as well as the unitaries Ui

		Psi = U1 U2 ... Ul X

		Returns X, U, S, the latter two as lists of arrays

	"""
    l = Psi.ndim

    S = []
    U = []

    #TODO - probably SVD is more accurate than eigh here??
    for j in range(l):  #For each leg

        if mode == 'eigh':
            rho = Psi.reshape((Psi.shape[0], -1))
            rho = np.dot(rho,
                         rho.T.conj())  #Compute density matrix of first leg
            p, u = np.linalg.eigh(rho)
            perm = np.argsort(-p)
            p = p[perm]
            p[p < 0] = 0.
            u = u[:, perm]
            Psi = np.tensordot(u.conj(), Psi, axes=[[0],
                                                    [0]])  #Strip off unitary
            s = np.sqrt(p)

        else:
            shp = Psi.shape
            Psi = Psi.reshape((shp[0], -1))
            u, s, v = svd(Psi, full_matrices=False)
            Psi = (v.T * s).T
            Psi = Psi.reshape((-1, ) + shp[1:])

        S.append(s)
        U.append(u)
        Psi = np.moveaxis(Psi, 0, -1)

    return Psi, U, S
#------------------------------------------------------------
def random_diag(l):
    return np.diag(2*np.random.randint(0,2,l)-1)
#------------------------------------------------------------
def svd_theta_UsV(theta, eta, p_trunc=0., verbose=False, random=False):
    """
    SVD of matrix, and resize + renormalize to dimension eta

    Returns: U, s, V, eta_new, p_trunc
        with s rescaled to unit norm
        p_trunc =  \sum_{i > cut}  s_i^2, where s is Schmidt spectrum of theta, REGARDLESS of whether theta is normalized
    """

    U, s, V = svd(theta, compute_uv=True, full_matrices=False)
    if random:
        #A=UsV=(U*D)s(D^\dag V), where D=diag(e^(i*phi1), e^(i*phi2),...)
        D = random_diag(len(s))
        U = U @ D
        V = D.conj() @ V

    pcum = np.cumsum(s**2)
    #assert(np.isclose(pcum[-1], 1., rtol=1e-8))
    ## This assertion is made because if nrm is not equal to 1.,
    ## the report truncation error p_trunc should be normalized?

    if p_trunc > 0.:
        eta_new = np.min([np.count_nonzero((1. - pcum/pcum[-1]) > p_trunc) + 1, eta])
    else:
        eta_new = np.min([eta, len(s)])

    return U[:, :eta_new], s[:eta_new] / np.sqrt(pcum[eta_new-1]), V[:eta_new, :], len(
        s[:eta_new]), pcum[-1] - pcum[eta_new-1]
#------------------------------------------------------------
def svd_theta(theta, truncation_par=None, form='A-SB', verbose=False):
    """ SVD and truncate a matrix based on truncation_par = {'chi_max': chi, 'p_trunc': p }

		Returns  normalized A, sB even if theta was not normalized

		info = {
		p_trunc =  \sum_{i > cut}  s_i^2, where s is Schmidt spectrum of theta, REGARDLESS of whether theta is normalized
    """
    if truncation_par is None:
        truncation_par = {}

    U, s, V = svd(theta, compute_uv=True, full_matrices=False)

    nrm = np.linalg.norm(s)
    if truncation_par.get('p_trunc', 0.) > 0.:
        eta_new = np.min([
            np.count_nonzero(
                (nrm**2 - np.cumsum(s**2)) > truncation_par.get('p_trunc', 0.))
            + 1,
            truncation_par.get('chi_max', len(s))
        ])
    else:
        eta_new = truncation_par.get('chi_max', len(s))
    nrm_t = np.linalg.norm(s[:eta_new])
    if form == 'A-SB':
        left = U[:, :eta_new]
        right = ((V[:eta_new, :].T) * s[:eta_new] / nrm_t).T
        #Here the * is element-wise multiplication, and shape broadcasting is used
        #diagflat(a,b) @ ((c,d),(e,f)) = ((a,0),(0,b)) @ ((c,d),(e,f)) = ((ac,ad),(be,bf))
        #=((c,d),(e,f))*((a,a),(b,b)) = ((c,d),(e,f))*(a,b) [with broadcasting]
    elif form == 'AS-B':
        left = U[:, :eta_new] * s[:eta_new]/nrm_t
        right = V[:eta_new, :]
    info = {
        'p_trunc': nrm**2 - nrm_t**2,
        's': s,
        'nrm': nrm,
        'eta': eta_new,
        'entropy': renyi(s, 1)
    }

    return left, right, info
#------------------------------------------------------------
def renyi(s, n):
    """n-th Renyi entropy from Schmidt spectrum s
    """
    s = s[s > 1e-16]
    if n == 1:
        return -2 * np.inner(s**2, np.log(s))
    elif n == 'inf':
        return -2 * np.log(s[0])
    else:
        return np.log(np.sum(s**(2 * n))) / (1 - n)
#------------------------------------------------------------
def Sn(psi, n = 1, group = [[0], [1]]):
    """ Given wf. psi, returns spectrum s & nth Renyi entropy via SVD
        group indicates how indices should be grouped into two parties
    """
    theta, pipe = group_legs(psi, group)
    s = svd(theta, compute_uv=False)
    S = renyi(s, n)
    return s, S
#----------------------------------------------------------------
def random_isometry(shape, contract_axes, dtype=np.float64):
    rand_matrix = np.random.random(shape)
    if dtype in [np.complex, np.complex64, np.complex128]:
        rand_matrix = rand_matrix + 1j * np.random.random(shape)
    Q, R = tensor_qr(rand_matrix, contract_axes)
    return Q
#-------------------------------------------------------------------------------
def get_QR_limit(psi, tol=1e-15, verbose=0):
    '''
      ---0   1---psi---2     ---
      |           |          |
      l           |       =  l
      |           |          |
      ---1   1---psi---2     ---

    return S, where SS^dag = l
    '''
    if not isinstance(psi, list):
        psi = [psi]
    assert psi[0].shape[1] == psi[-1].shape[2]

    N = len(psi)
    D = psi[0].shape[1]
    try:
        def mv(l):
            l = l.reshape([D,D])
            for n in range(N):
                l_psi = np.tensordot(l, psi[n], [[0],[1]])
                l = np.tensordot(l_psi, psi[n].conj(), [[0,1],[1,0]])
            return l.reshape([D**2])
        LinOp = sp.sparse.linalg.LinearOperator((D**2, D**2), matvec=mv)
        w, v = sp.sparse.linalg.eigs(LinOp, k=1, which='LM',
            tol=tol, return_eigenvectors=True)
    except:
        print("exception in get_QR_limit")
        Ts = [np.tensordot(A, A.conj(), [[0],[0]]) for A in psi]
        Ts = [group_legs(T, [[1,3],[0,2]])[0] for T in Ts]
        T = Ts[0]
        for n in range(1, N):
            T = T @ Ts[n]
        w, v = sp.sparse.linalg.eigs(T, k=1, which='LM',
                tol=tol, return_eigenvectors=True)
    #w, v = np.linalg.eig(T)
    eig_T = sorted(w, key=abs, reverse=True)
    sort_index = np.argsort(np.absolute(w))[::-1]
    l = ((v.T)[sort_index]).T[:,0]
    l = l.reshape([psi[0].shape[1], psi[0].shape[1]])

    max_element = np.diag(l)[abs(np.diag(l)).argmax()]
    phase = np.exp(cmath.phase(max_element)*1.j)
    l = l/phase

    #When T has degenerate leading eigenvalues, l1 = S1S1^\dag, l2 = S2S2^\dag, and
    #the returned eigenvector l may be some combination of l1 and l2:
    #l= a*l1+b*l2 not= S S^dag for any S
    try:
        S = np.linalg.cholesky(l).T
        return S/np.linalg.norm(S), np.sqrt(eig_T[0].real)
    except:
        print("Add small I to do Cholesky in get_QR_limit")
        try:
            l = l + np.eye(l.shape[0])*1e-14
            S = np.linalg.cholesky(l).T
            return S/np.linalg.norm(S), np.sqrt(eig_T[0].real)
        except:
            print("cholesy failed...return a random S")
            return normalized_random(l.shape), np.sqrt(eig_T[0].real)
        #eig_T[0] should be real, even for complex psi.


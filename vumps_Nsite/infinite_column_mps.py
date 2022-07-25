import pickle
import numpy as np
import scipy as sp
import scipy.linalg
from misc import * 
import cmath
import time

class iColumnMPS:
    """
    The column MPS class.
    Differs from the usual MPS because there may be multiple physical indices
    """
#-------------------------------------------------------------------------------
    def __init__(self, tensors=None):
        if tensors is not None:
            self.Psi = tensors.copy()
        else:
            self.Psi = None
#-------------------------------------------------------------------------------
    def copy(self):
        """ make a copy of self at a different memory location """
        icmps1 = iColumnMPS()
        icmps1.Psi = self.Psi.copy()
        return icmps1
#-------------------------------------------------------------------------------
    def TB_invert(self):
        """
        Top-bottom inversion of Psi
        A iColumnMPS in A-form is mapped into a iColumnMPS with B-form.
        """
        self.Psi = [A.transpose([0,2,1]) for A in self.Psi[::-1]]
#-------------------------------------------------------------------------------
    def to_form(self, form='A', max_iter=100, epsilon=1e-8, truncation_par=None, verbose=-1, init='random'):
        """
        Put the mps to A-form (arrows pointing up) or B-form (arrows pointing down). 
        No need to be in canonical form at first. 
        Forces normalization.
        """
        assert self.Psi[0].ndim == 3
        
        t0 = time.time()
        if form == 'B':
            self.TB_invert() 

        if verbose >= 0:
            print("-"*40)
            print("to_form        :", form)
            print("to_form input  :", self.Psi[0].shape)
            get_gap(self.Psi[0], phys_inds=[0], st="before to_form") 


        N = len(self.Psi)
        #1--psi--2
        #    |
        #    0
        Cs = [None] * N
        ALs = [None] * N
        ACs= [None] * N
        if init == 'random':
            Cs[-1] = normalized_random([self.Psi[-1].shape[2], self.Psi[0].shape[1]])
        elif init == 'QR':
            Cs[-1], lam = get_QR_limit(self.Psi, tol=1e-14)
            self.Psi[0] = self.Psi[0]/lam

        diff = 1
        j = 0
        go = True
        while j < max_iter and diff > epsilon:
            C_last = Cs[-1]
            for n in range(N):
                ACs[n] = np.tensordot(Cs[(n-1)%N], self.Psi[n], [[1],[1]]).transpose([1,0,2])
                ACs[n] = ACs[n]/np.linalg.norm(ACs[n]) 

                AC_mat, pipe = group_legs(ACs[n], [[0,1], [2]]) 
                AL_mat, Cs[n] = unique_qr(AC_mat)
                ALs[n] = ungroup_legs(AL_mat, pipe)
                Cs[n] = Cs[n]/np.linalg.norm(Cs[n])

                AL_C = np.tensordot(ALs[n], Cs[n], [[2],[0]])
                AC   = ACs[n]
            if Cs[-1].shape == C_last.shape:
                diff = np.linalg.norm(Cs[-1]-C_last)
            j += 1
            if verbose >= 1:
                print("-"*20)
                print("to form iter  :", j) 
                print("dC        :", diff)

        self.Psi = ALs
        if form == 'B':
            self.TB_invert()
            ACs = [A.transpose([0,2,1]) for A in ACs[::-1]]
            Cs  = [C.transpose([1,0]) for C in Cs[::-1]]
            Cs.append(Cs.pop(0))

        if verbose >= 0:
            print("to_form iter   :", j)
            print("to_form dpsi_C :", diff)
            print("to_form time   :", time.time()-t0)
            get_gap(self.Psi[0], phys_inds=[0], st="after to_form") 


        info = {'Cs'    : Cs,
                'ACs'   : ACs, 
                'iter'  : j, 
                'diff'  : diff, 
               }
        return info


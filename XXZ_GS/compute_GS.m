clear;
clc;
close all;


%% parameters
%   H = -\sum_j (Sx_{j} Sx_{j+1} +  Sy_{j} Sy_{j+1} + Delta Sz_{j} Sz_{j+1}) - h \sum_j Sz_{j}

Delta   = -0.2;
h       = 0;
tol     = 1e-12;
verbose = 1; 
assert(abs(h)<1-Delta,'|h|<1-Delta! otherwise trivial FM!');

opts    = optimset('TolX',tol);
dig     = ceil(-log10(tol));
frmt    = ['%2.',int2str(dig),'e'];

[E0,rho,B,Edr,m0]=fXXZGS_fixedh(Delta,h,tol,verbose); 
disp(['E0=',num2str(E0,frmt),', m0=',num2str(m0,frmt), 'B=',num2str(B,frmt)]);

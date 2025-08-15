from fenics import *
from dolfin import *
from numpy import loadtxt
N_mode = 40
Eig = loadtxt("./PODmode/Eigenvalue/Eig_core_level_eq.txt")
for n_t in range(0,N_mode):
    E_t = 0
    E_pod = 0
    for i in range(0, len(Eig)):
        if Eig[i] > 0:
            E_t += Eig[i]
            if i > n_t:
                E_pod += Eig[i]
    Err_file = open('theoretical_error_Core_level.txt','a')
    Err_file.write('%.16g\n' % (100*sqrt(E_pod/E_t)))
    Err_file.close()

# fenics code for the thermal simulation of chip for the publication  for frequency is 3.5 GHZ
from fenics import *
from dolfin import * 
from mshr import *
from numpy import loadtxt
from petsc4py import PETSc
import numpy as np
import sys
import time
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
#define the parameter
N_STATE = 0       
num_steps = 3000                                                  # num_steps is the number of time steps
Nu = 404                                                         # Nu is the number if functional unit
thick_actl =1.35e-4      
N_mode = 10
flp = loadtxt('gv100_flp.txt')
pd = loadtxt('power_GV100_pre1.txt')                                        #pd  is power density
#compute power desity
for i in range(0,num_steps):
    for j in range(0,Nu):
        pd[i,j] = pd[i,j]/(flp[j,0]*flp[j,1]*thick_actl)
for Nu2 in range(0,Nu):
    mode_inte_name = './PODmode/P_matrix_'+str(Nu2)+'.txt'
    mode_inte = loadtxt(mode_inte_name)
    P_matrix = np.zeros((N_mode,num_steps))
    for j in range(0, N_mode):
        for i in range(0, num_steps):
            P_matrix[j][i]=1.9*pd[i,Nu2]*mode_inte[j]
    header = './Power_pre1_result/P_matrix_pre1_'+str(Nu2)
    P_matrix_file_name = header + '.txt'
    P_matrix_file = open(P_matrix_file_name,'w')
    for k1 in range(0,N_mode):
        for k2 in range(0,num_steps):
            P_matrix_file.write('%.16g\t' % (P_matrix[k1][k2]))
        P_matrix_file.write('\n')
    P_matrix_file.close()


# fenics code for the thermal simulation of chip for the publication  for frequency is 3.5 GHZ
from fenics import *
from dolfin import * 
from mshr import *
from numpy import loadtxt
from petsc4py import PETSc
import numpy as np
import time
import sys
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
#define the parameter
N_STATE = 1                                                         # N_STATE is the controlling parameter   N_STATE = 0 indicates that we need to solve PDE and generate data; 	N_STATE = 1 indicate that we already have data just need to read it.													# w is the width of domain m
h = 7.2e-4															# h is the thickness of domain m												# hs is number of grid along with h direction
T = 3*2000*5.33e-6                                                       # T is final time or total time (s)
num_steps = 2000                                                  # num_steps is the number of time steps
Train_steps = num_steps
t = 0
dt = T/num_steps                                                # time step size 
Rb = 2.26                                                     # the thermal resistor of convectional face
#ch = 1/(Rb*Ach)                                                         # ch is convective coefficient
h_c = 2.40598e4
Ta = 0                                                         # Ta is initial /ambient temperature                                                           # Nu is the number if functional unit
thick_actl = 1.35e-4                                            # lthick_actl is the total thickness of active layer (m)                                           #chip_area is the area of chip
#Line Format: <unit-name>\t<width>\t<height>\t<left-x>\t<bottom-y>\t[<specific-heat>]\t[<resistivity>]
#compute power desity
#create geometric model
#Nu2 = int(sys.argv[1])
ls = 207
ws = 278
hs = 17
mesh = BoxMesh(Point(0,0,0), Point(0.0088,0.01179375,7.2e-4 ),ls-1,ws-1,hs-1)
coor1 = mesh.coordinates()
lr = coor1[:,0].max()
ll = coor1[:,0].min()
wb = coor1[:,1].min()
wt = coor1[:,1].max()
hmax = coor1[:,2].max()
hmin = coor1[:,2].min()
V = FunctionSpace(mesh, 'P', 1)


#define thermal conductivity 
#define thermal conductivity 
tol = 1E-14
k_0 = 100                                                                                                               # silicon conductivity      (w/(m.k))
k_1 = 1.2                                                   #oxide silicon thermal conductivity
kappa = Expression('x[2] <= 0.000242 + tol ? k_0 : k_1', degree=0,tol=tol, k_0=k_0, k_1=k_1) #define subdomain
#define density 
D0 = 2.33e3                                                         # silicon density    (kg/m^3)
D1 = 2.65e3                                                        # oxide silicon density
DS1 = Expression('x[2] <= 0.000242 + tol ? D0 : D1', degree=0,tol=tol, D0=D0, D1=D1) #define subdomain
#define specific heat
c0 = 751.1                                                         # silicon specific heat   (J/(kg.k))
c1 = 680                                                         # oxide silicon specific heat
sc = Expression('x[2] <= 0.000242 + tol ? c0 : c1', degree=0,tol=tol, c0=c0, c1=c1) #define subdomain

T_integral = 0
u0 = Constant(Ta)                                                                   # Ta is initial temperature 
u_n = interpolate(u0,V)
# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
solution = []
for n in range(0,Train_steps):
	solution.append('u1')
# Sum integrals to define variational problem
#a = DS1*sc*u*v*dx + dt*kappa*dot(grad(u), grad(v))*dx + sum(integrals_R_a)
coor = mesh.coordinates()
v2d = vertex_to_dof_map(V)
# if we have the solutiuon, we just need to  Load solution# #######################read the solution from a solution file ######################
u = Function(V)
for n in range(0,Train_steps):
    solution_load_file_name = "solution_Core_train/file_" + str(n) + "h5"
    #solution_load_file_name = "solution_"+str(Nu2)+"_outof_13/file_" + str(n) + "h5"
    solution_file = HDF5File(mesh.mpi_comm(), solution_load_file_name, "r")
    solution_file.read(u, "solution")
    solution[n] = interpolate(u0,V)
    solution[n].assign(u)
    solution_file.close()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~generate podmode~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ################# generate the data matrix#####################
# ----write coorelation matrix into a text file-------
mm = np.zeros((Train_steps,Train_steps)) 
header = 'corelationmatrix_Core'                               #define the header of node value file
corelationmatrix_file_name = header + '.txt'
corelationmatrix_file = open(corelationmatrix_file_name,'w')
#co_matrix = PETScMatrix()   # co_matrix is correlation matrix
#co_matrix.mat().setSizes([Train_steps,Train_steps])
#co_matrix.mat().setType("dense")
#co_matrix.mat().setUp()
for i in range(0,Train_steps):
    for j in range(0,Train_steps):
        if i <= j:
            mm[i][j] = assemble(dot(solution[i],solution[j])*dx)/Train_steps
            #co_matrix.mat().setValues(i,j,mm[i][j])
            corelationmatrix_file.write('%.16g\t' % (mm[i][j]))
        else :
            #co_matrix.mat().setValues(i,j,mm[j][i])
            corelationmatrix_file.write('%.16g\t' % (mm[j][i]))
    corelationmatrix_file.write('\n')
#co_matrix.mat().assemble()
corelationmatrix_file.close()

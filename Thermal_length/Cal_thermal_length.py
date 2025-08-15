# fenics code for the thermal simulation of chip for the publication  for frequency is 3.5 GHZ
from fenics import *
from dolfin import * 
from mshr import *
from numpy import loadtxt
from petsc4py import PETSc
import numpy as np
import time
from mpi4py import MPI
import sys
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
#define the parameter
h_mm = int(sys.argv[1])
l_p_mm = int(sys.argv[3])
w_p_mm = int(sys.argv[2])
h = h_mm/1e6
l_p = l_p_mm/1000
w_p = w_p_mm/1000
l = 0.0286
w = 0.0285
N_STATE = 1                                                         # N_STATE is the controlling parameter   N_STATE = 0 indicates that we need to solve PDE and generate data; 	N_STATE = 1 indicate that we already have data just need to read it.													# w is the width of domain m
T = 600*3.125e-6                                                          # T is final time or total time (s)
num_steps = 600                                                  # num_steps is the number of time steps
Train_steps = num_steps
t = 0
dt = T/num_steps                                                # time step size 
Rb = 2.26                                                     # the thermal resistor of convectional face
#ch = 1/(Rb*Ach)                                                         # ch is convective coefficient
h_c = 2.40598e4
Ta = 0                                                         # Ta is initial /ambient temperature                                                           # Nu is the number if functional unit
thick_actl = (1.35/7.2)*h                                            # lthick_actl is the total thickness of active layer (m)                                           #chip_area is the area of chip
#Line Format: <unit-name>\t<width>\t<height>\t<left-x>\t<bottom-y>\t[<specific-heat>]\t[<resistivity>]
#compute power desity
#create geometric model
#mesh_file = XDMFFile(comm, "./buidling_blk_9/Block9.xdmf")
ls = 673
ws = 673
hs = 20
mesh = BoxMesh(Point(0,0,0), Point(l,w,h),ls-1,ws-1,hs-1)
V = FunctionSpace(mesh, 'P', 1)
coor = mesh.coordinates()

lr = coor[:,0].max()
ll = coor[:,0].min()
wb = coor[:,1].min()
wt = coor[:,1].max()
hmax = coor[:,2].max()
hmin = coor[:,2].min()
Num_nodes = mesh.num_vertices()
#V = FunctionSpace(mesh, 'P', 1)
v2d = vertex_to_dof_map(V)


#define thermal conductivity 
tol = 1E-14
k_0 = 100														# silicon conductivity      (w/(m.k))
k_1 = 100                                                      #oxide silicon thermal conductivity
kappa = Expression('x[2] <= 0.00045 + tol ? k_0 : k_1', degree=0,tol=tol, k_0=k_0, k_1=k_1) #define subdomain
#define density 
D0 = 2.33e3                                                         # silicon density    (kg/m^3)
D1 = 2.33e3                                                        # oxide silicon density
DS1 = Expression('x[2] <= 0.00045 + tol ? D0 : D1', degree=0,tol=tol, D0=D0, D1=D1) #define subdomain
#define specific heat
c0 = 751.1                                                         # silicon specific heat   (J/(kg.k))
c1 = 751.1                                                         # oxide silicon specific heat
sc = Expression('x[2] <= 0.00045 + tol ? c0 : c1', degree=0,tol=tol, c0=c0, c1=c1) #define subdomain

T_integral = 0
u0 = Constant(Ta)                                                                   # Ta is initial temperature 
u_n = interpolate(u0,V)
# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
solution = []
for n in range(0,1):
	solution.append('u1')
# Sum integrals to define variational problem
#a = DS1*sc*u*v*dx + dt*kappa*dot(grad(u), grad(v))*dx + sum(integrals_R_a)
coor = mesh.coordinates()
v2d = vertex_to_dof_map(V)
# if we have the solutiuon, we just need to  Load solution# #######################read the solution from a solution file ######################
u = Function(V)
x_1 = 0
x_2 = 0
solution_file_name = "static_Temp/T_"+str(h_mm) +"_" + str(l_p_mm)+"_"+str(w_p_mm)+"h5"
solution_file = HDF5File(mesh.mpi_comm(), solution_file_name, "r")
solution_file.read(u, "solution")
solution_file.close()
Max = u((0.014,0.014,h-thick_actl))
Min = u((0.014 +l_p/2,0.014,h-thick_actl))
y1 = np.linspace(0.014 + tol, l- tol, 700)
point1 = [(y_, 0.014,h-thick_actl) for y_ in y1]
data = np.array([u(point) for point in point1])
for i in range(0, len(y1)):
    if abs(data[i]-0.37*Max) < 1e-5:
        x_1 = 1000*(y1[i] - 0.014)
for i in range(0, len(y1)):
    if abs(data[i]-0.37*Min) < 1e-5:
        x_2 = 1000*(y1[i] - 0.014 - (l_p/2))

data_file_name = "thermal_lth_"+str(h_mm)+"_"+str(w_p_mm)+".txt"  
data_file = open(data_file_name,'a') 
data_file.write('%.16g\t' % (x_1))
data_file.write('%.16g\t' % (x_2))
data_file.write('%.16g\n' % ((x_1+x_2)/2))
data_file.close()


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
Nu2 = 0#int(sys.argv[1])# N_STATE is the controlling parameter   N_STATE = 0 indicates that we need to solve PDE and generate data; 	N_STATE = 1 indicate that we already have data just need to read it.
parms = parameters["krylov_solver"]
parms["relative_tolerance"]=1e-23
parms["absolute_tolerance"]=1e-25
####to generate the simulation mesh
ls = 207
ws = 278 
hs = 17
mesh = BoxMesh(Point(0,0,0), Point(0.0088,0.01179375,7.2e-4),ls-1,ws-1,hs-1)
coor1 = mesh.coordinates()
lr = coor1[:,0].max()
ll = coor1[:,0].min()
wb = coor1[:,1].min()
wt = coor1[:,1].max()
hmax = coor1[:,2].max()
hmin = coor1[:,2].min()
T = 2000*3.125e-6                                                          # T is final time or total time (s)
V = FunctionSpace(mesh, 'P', 1)


num_steps = 2000                                                  # num_steps is the number of time steps
t = 0
dt = T/num_steps                                                # time step size 
Rb = 2.26                                                     # the thermal resistor of convectional face
#ch = 1/(Rb*Ach)                                                         # ch is convective coefficient
h_c = 2.40598e4
#h_s = 1.40598e2
Ta = 0                                                         # Ta is initial /ambient temperature 
Nu = 1                                                         # Nu is the number if functional unit
thick_actl =1.35e-4      
N_mode = 20
# lthick_actl is the total thickness of active layer (m)
#import power trace file
#import floorplan file
#Line Format: <unit-name>\t<width>\t<height>\t<left-x>\t<bottom-y>\t[<specific-heat>]\t[<resistivity>]
flp = loadtxt('Core_flp.txt')
#compute power desity 
#create mesh and define function space 

pd = loadtxt('power_SM_training.txt')           
#print(pd[0,0])
#pd  is power density
#import floorplan file
#Line Format: <unit-name>\t<width>\t<height>\t<left-x>\t<bottom-y>\t[<specific-heat>]\t[<resistivity>]
#compute power desity
for i in range(0,num_steps):
    for j in range(0,Nu):
        #print(j)
        pd[i] = pd[i]/(flp[0]*flp[1]*thick_actl)
#define thermal conductivity 
#define thermal conductivity 
tol = 1E-14
k_0 = 100                                                                                                               # silicon conductivity      (w/(m.k))
k_1 = 1.2                                                   #oxide silicon thermal conductivity
kappa = Expression('x[2] <= 5.85e-4 + tol ? k_0 : k_1', degree=0,tol=tol, k_0=k_0, k_1=k_1) #define subdomain
#define density 
D0 = 2.33e3                                                         # silicon density    (kg/m^3)
D1 = 2.65e3                                                        # oxide silicon density
DS1 = Expression('x[2] <= 5.85e-4 + tol ? D0 : D1', degree=0,tol=tol, D0=D0, D1=D1) #define subdomain
#define specific heat
c0 = 751.1                                                         # silicon specific heat   (J/(kg.k))
c1 = 680                                                         # oxide silicon specific heat
sc = Expression('x[2] <= 5.85e-4 + tol ? c0 : c1', degree=0,tol=tol, c0=c0, c1=c1) #define subdomain


# Define boundary subdomains
class source_term(UserExpression):
	def _init_(self,Nu2,flp,dd,hmax,thick_actl,pd,**kwargs):
		self.Nu2, self.flp, self.dd,self.hmax,self.thick_actl,self.pd= Nu2,flp,dd,hmax,thick_actl,pd
	def eval(self,value,x):
		tol = 1e-14
		if x[2] > 5.85e-4 - thick_actl and x[2]<5.85e-4:
                    if x[0] >= flp[2] and x[0] < (flp[0] + flp[2] +tol) and x[1] >= flp[3] and x[1] < (flp[3] +flp[1] + tol):
                    #if x[0] >= flp[Nu2][2] and x[0] < (flp[Nu2][0] + flp[Nu2][2] +tol) and x[1] >= flp[Nu2][3] and x[1] < (flp[Nu2][3] +flp[Nu2][1] + tol):
                        value[0] = 1.0
                    else:
                        value[0] = 0.0
			
		else:
			value[0] = 0.0
	def value_shape(self):
		return ()



class BoundaryX0(SubDomain):
      def inside(self, x, on_boundary):
          return on_boundary and near(x[0], ll, tol)

class BoundaryX1(SubDomain):
      def inside(self, x, on_boundary):
          return on_boundary and near(x[0], lr, tol)

class BoundaryY0(SubDomain):
      def inside(self, x, on_boundary):
          return on_boundary and near(x[1], wb, tol)

class BoundaryY1(SubDomain):
      def inside(self, x, on_boundary):
          return on_boundary and near(x[1], wt, tol)
class BoundaryZ0(SubDomain):
      def inside(self, x, on_boundary):
          return on_boundary and near(x[2], hmax, tol)

class BoundaryZ1(SubDomain):
      def inside(self, x, on_boundary):
          return on_boundary and near(x[2], hmin, tol)
boundary_markers = MeshFunction('size_t', mesh,2)
boundary_markers.set_all(9999)
bx0 = BoundaryX0()
bx1 = BoundaryX1()
by0 = BoundaryY0()
by1 = BoundaryY1()
bz0 = BoundaryZ0()
bz1 = BoundaryZ1()
bx0.mark(boundary_markers, 0)
bx1.mark(boundary_markers, 1)
by0.mark(boundary_markers, 2)
by1.mark(boundary_markers, 3)
bz0.mark(boundary_markers, 4)
bz1.mark(boundary_markers, 5)
# Redefine boundary integration measure
ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)
T_integral = 0
# Define boundary conditions
boundary_conditions = {0: {'Neumann': 0},   # x = 0 adiabatic boundary condition
                       1: {'Neumann': 0},   # x = 1
                       2: {'Neumann': 0},   # y = 0
                       3: {'Neumann': 0},    # y = 1
		       4: {'Neumann': 0}, # z = 0
		       5: {'Robin': (h_c, Ta)}}      # z = 1       r is dt*h , s = Ta reference temperature
					   #5:{'Robin': (dt*k_0, Ta)}}      # z = 1       r is dt*h , s = Ta reference temperature

# Collect Dirichlet conditions
bcs = []
for i in boundary_conditions:
	if 'Dirichlet' in boundary_conditions[i]:
		bc = DirichletBC(V, boundary_conditions[i]['Dirichlet'], boundary_markers, i)
		bcs.append(bc) 
#define initial value 
u0 = Constant(Ta)                                                                   # Ta is initial temperature 
u_n = interpolate(u0,V)
# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
# Collect Neumann integrals
integrals_N = []
for i in boundary_conditions:
    if 'Neumann' in boundary_conditions[i]:
        if boundary_conditions[i]['Neumann'] != 0:
            g = boundary_conditions[i]['Neumann']
            integrals_N.append(g*v*ds(i))	
# Collect Robin integrals
integrals_R_a = []
integrals_R_L = []
for i in boundary_conditions:
	if 'Robin' in boundary_conditions[i]:
		r, s = boundary_conditions[i]['Robin']
		integrals_R_a.append(r*u*v*ds(i))
		integrals_R_L.append(r*s*v*ds(i))			
podmode = []
for n in range(0,N_mode):
    podmode.append('u1')
u = Function(V)
for n in range(0,N_mode):
    mode_load_file_name = "podmode_Core/mode_" + str(n) + "h5"
    mode_file = HDF5File(mesh.mpi_comm(), mode_load_file_name, "r")
    mode_file.read(u, "solution")
    podmode[n] = interpolate(u0,V)
    podmode[n].assign(u)
    mode_file.close()
dd = 2
f = source_term(Nu2, flp,dd,hmax,thick_actl, pd)
P_matrix = np.zeros((N_mode,1))
for j in range(0, N_mode):
    value_P = assemble(dot(f,podmode[j])*dx)
    P_matrix[j][0] = value_P
    print(j)
header = 'P_matrix_Core'
P_matrix_file_name = header + '.txt'
P_matrix_file = open(P_matrix_file_name,'w')
for k1 in range(0,N_mode):
    P_matrix_file.write('%.16g\n' % (P_matrix[k1][0]))
P_matrix_file.close()


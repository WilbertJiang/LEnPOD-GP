# fenics code for the thermal simulation of chip for the publication
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
N_STATE = 0                                                        # N_STATE is the controlling parameter   N_STATE = 0 indicates that we need to solve PDE and generate data; 	N_STATE = 1 indicate that we already have data just need to read it.
h_mm = int(sys.argv[1])
l_p_mm = int(sys.argv[2])
w_p_mm = int(sys.argv[3])
h = h_mm/1e6
l_p = l_p_mm/1000
w_p = w_p_mm/1000
l = 0.0286															# l is the length of domain m
w = 0.0285															# w is the width of domain m
ls = 673															# ls is number of grid along with l direction
ws = 673															# ws is number of grid along with w direction
hs = 20														# hs is number of grid along with h direction
t = 0
Rb = 0.1075695    
# the thermal resistor of convectional face
#ch = 1/(Rb*Ach)                                                         # ch is convective coefficient
h_c = 2.40598e4
Ta = 0                                                         # Ta is initial /ambient temperature 
Nu = 13                                                          # Nu is the number if functional unit
num_steps = 1
thick_actl = (1.35/7.2)*h                                        # lthick_actl is the total thickness of active layer (m)
chip_area = l*w                                                  #chip_area is the area of chip
#import power trace file                                        #pd  is power density
#create geometric model
M = Box(Point(0,0,0), Point(l,w,h))
mesh = BoxMesh(Point(0,0,0), Point(l,w,h),ls-1,ws-1,hs-1)
V = FunctionSpace(mesh, 'P', 1)
#define thermal conductivity 
tol = 1E-14
k_0 = 100														# silicon conductivity      (w/(m.k))
k_1 = 1.2                                                      #oxide silicon thermal conductivity
kappa = Expression('x[2] <= h - thick_actl + tol ? k_0 : k_1', degree=0,h=h,thick_actl=thick_actl,tol=tol, k_0=k_0, k_1=k_1) #define subdomain
#define density 
D0 = 2.33e3                                                         # silicon density    (kg/m^3)
D1 = 2.65e3                                                        # oxide silicon density
DS1 = Expression('x[2] <= h - thick_actl + tol ? D0 : D1', degree=0,h=h,thick_actl=thick_actl,tol=tol, D0=D0, D1=D1) #define subdomain
#define specific heat
c0 = 751.1                                                         # silicon specific heat   (J/(kg.k))
c1 = 680                                                         # oxide silicon specific heat
sc = Expression('x[2] <= h - thick_actl  + tol ? c0 : c1', degree=0,h=h,thick_actl=thick_actl,tol=tol, c0=c0, c1=c1) #define subdomain
#define power source term 
class source_term(UserExpression):
    def _init_(self,h,thick_actl,l_p,**kwargs):
        self.h,self.thick_actl,self.l_p,self.w_p = h,thick_actl,l_p,w_p
    def eval(self,value,x):
        tol = 1e-14
        if x[2] < h - thick_actl and x[2] > h - 2*thick_actl:
            if x[0] >= 0.014 -l_p/2 and x[0] < (0.014 +l_p/2+tol) and x[1] >= 0.014 - w_p/2 and x[1] < (0.014+w_p/2 + tol):
                value[0] = 1e6		
            else:
                value[0] = 0.0
        else:
            value[0] = 0.0
    def value_shape(self):
        return ()
# Define boundary condition
# Define boundary subdomains
class BoundaryX0(SubDomain):
      def inside(self, x, on_boundary):
          return on_boundary and near(x[0], 0, tol)

class BoundaryX1(SubDomain):
      def inside(self, x, on_boundary):
          return on_boundary and near(x[0], l, tol)

class BoundaryY0(SubDomain):
      def inside(self, x, on_boundary):
          return on_boundary and near(x[1], 0, tol)

class BoundaryY1(SubDomain):
      def inside(self, x, on_boundary):
          return on_boundary and near(x[1], w, tol)
class BoundaryZ0(SubDomain):
      def inside(self, x, on_boundary):
          return on_boundary and near(x[2], h, tol)

class BoundaryZ1(SubDomain):
      def inside(self, x, on_boundary):
          return on_boundary and near(x[2], 0, tol)
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
					   #5:{'Dirichlet': 0}}      # z = 1       r is dt*h , s = Ta reference temperature
		       5:{'Robin': (h_c, Ta)}}      # z = 1       r is dt*h , s = Ta reference temperature

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
solution = []
for n in range(0,num_steps):
	solution.append('u1')
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

# Sum integrals to define variational problem
#a = DS1*sc*u*v*dx + dt*kappa*dot(grad(u), grad(v))*dx + sum(integrals_R_a)
coor = mesh.coordinates()
v2d = vertex_to_dof_map(V)
if N_STATE == 0:
    a =  kappa*dot(grad(u), grad(v))*dx +sum(integrals_R_a)#(((k_0*u)/h )/(k_0*chip_area/h + 1/(Rb)))/(Rb*chip_area)*v*ds(5)
    #a = DS1*sc*u*v*dx + dt*kappa*dot(grad(u), grad(v))*dx + k_0*u*dt*v*ds(5)
    # Compute solution
    u = Function(V)
    #header = 'matrix_fenics/nodevalue1'                               #define the header of node value file
    #value_file_name = header + '.txt'
    dd = 1
    f = source_term(h,thick_actl, l_p,w_p)# source_term(number_unit = Nu, floor_plan = flp,current_timestep = dd,total_thickness = h,power_thickness = layer_thickness, power_density = pd)
    L =  f*v*dx 
    #L =  f*v*dx - sum(integrals_N) +sum(integrals_R_L)
    #compute solution
    solve(a == L, u, bcs,solver_parameters={'linear_solver':'gmres'})
    u_nodal_values = u.vector()
    u_array = u_nodal_values.get_local()
    # save the solution into a file which can be import directly 
    solution_file_name = "static_Temp/T_"+str(h_mm) +"_" + str(l_p_mm)+"_"+str(w_p_mm)+"h5"
    solution_file = HDF5File(mesh.mpi_comm(), solution_file_name, "w")
    solution_file.write(u,"solution")
    solution_file.close()


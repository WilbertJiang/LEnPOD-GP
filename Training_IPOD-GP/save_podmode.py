# fenics code for the thermal simulation of chip for the publication  for frequency is 3.5 GHZ
#import meshio
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
#define the parameter													# w is the width of domain m
#Nu2 = int(sys.argv[1])
ls = 207
ws= 278
hs = 17
mesh = BoxMesh(Point(0,0,0), Point(0.0088,0.01179375,7.2e-4),ls-1,ws-1,hs-1)
V = FunctionSpace(mesh, 'P', 1)
coor1 = mesh.coordinates()
lr = coor1[:,0].max()
ll = coor1[:,0].min()
wb = coor1[:,1].min()
wt = coor1[:,1].max()
hmax = coor1[:,2].max()
hmin = coor1[:,2].min()
num_steps = 2000
Train_steps = 2000# num_steps is the number of time steps
t = 0
Rb = 2.26                                                     # the thermal resistor of convectional face
#ch = 1/(Rb*Ach)                                                         # ch is convective coefficient
h_c = 2.40598e4
Ta = 0                                                         # Ta is initial /ambient temperature 
N_mode = 50
Nu = 1                                                         # Nu is the number if functional unit
thick_actl = 1.35e-4                                       # lthick_actl is the total thickness of active layer (m)

Num_nodes = mesh.num_vertices()
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

#define power source term 

T_integral = 0
#define initial value 
u0 = Constant(Ta)                                                                   # Ta is initial temperature 
u_n = interpolate(u0,V)
# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
solution = []
for n in range(0,Train_steps):
	solution.append('u1')
# Collect Neumann integrals
#a = DS1*sc*u*v*dx + dt*kappa*dot(grad(u), grad(v))*dx + sum(integrals_R_a)
coor = mesh.coordinates()
v2d = vertex_to_dof_map(V)
# if we have the solutiuon, we just need to  Load solution# #######################read the solution from a solution file ######################
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

u = Function(V)
for n in range(0,Train_steps):
    solution_load_file_name = "solution_Core_train/file_" + str(n) + "h5"
    solution_file = HDF5File(mesh.mpi_comm(), solution_load_file_name, "r")
    solution_file.read(u, "solution")
    solution[n] = interpolate(u0,V)
    solution[n].assign(u)
    solution_file.close()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~generate podmode~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~generate podmode~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ################# generate the data matrix#####################
# ----write coorelation matrix into a text file-------

data_filename = 'corelationmatrix_Core.txt'
mm = loadtxt(data_filename)
co_matrix = PETScMatrix()   # co_matrix is correlation matrix
co_matrix.mat().setSizes([Train_steps,Train_steps])
co_matrix.mat().setType("dense")
co_matrix.mat().setUp()
for i in range(0,Train_steps):
	for j in range(0,Train_steps):
		if i <= j:
			co_matrix.mat().setValues(i,j,mm[i][j])
		else :
			co_matrix.mat().setValues(i,j,mm[j][i])
co_matrix.mat().assemble()
# ######################### solve eigenvalue and eigenvector############################
e_value_r = []
e_vec_r = []
# Create eigensolver
PETScOptions.set("st_ksp_type", "preonly")
PETScOptions.set("st_pc_type", "lu")
PETScOptions.set(" -pc_factor_mat_solver_type", "mumps")
eigensolver = SLEPcEigenSolver(co_matrix)
# Configure solver to get the largest eigenvalues first:
eigensolver.parameters["spectrum"] = "largest real"
# Compute all eigenvalues of A x = \lambda x
print ('Computing eigenvalues. This can take a minute')
print(' solve %d largest eigenpairs' %(Train_steps))
eigensolver.solve()
header = 'eigenvalue_r_Core'                              
eigenvalue_file_name = header + '.txt'
eigenvalue_file = open(eigenvalue_file_name,'w')
for i in range (0,Train_steps):
	# Extract largest (first) eigenpair
	r, c, rx , cx = eigensolver.get_eigenpair(i)
	e_value_r.append(r)
	e_vec_r.append(rx)
	eigenvalue_file.write('%.16g\n' % (r))
	#print(' print the eigenvalue \n')
	#print('%8g\n' %(r))
eigenvalue_file.close()
print('eigenvalue is done \n')
# ###########################################################################
# ###########################################################################
# ######################### generate the pod mode ############################
# ###########################################################################
# ###########################################################################
podmode = []

#podmode_vector_file = open('podmode_fenics_core1_test_square_lp_redo.txt','w')      # save the podmode into a file 
for i in range (0,55):
#for i in range (0,Train_steps):
    a_init = Constant(0)
    a = interpolate(a_init,V)
    for j in range (0, Train_steps):	
        a.vector().axpy(e_vec_r[i][j], solution[j].vector())
    a.vector()[:] =  a.vector()/(e_value_r[i]*Train_steps)
	# **********normalize the podmode ***************
    normalize_value = assemble(dot(a,a)*dx)
    a.vector()[:] =  a.vector()/sqrt(normalize_value)
    #print(assemble(dot(a,a)*dx))
    podmode.append(a)
    pod_filename = 'podmode_fenics_Core.txt'
    podmode_vector_file = open(pod_filename,'a')
    for n_count in range(0,Num_nodes):
        podmode_vector_file.write('%.16g\t' % (a.vector()[v2d[n_count]]))
    podmode_vector_file.write('\n')
    podmode_vector_file.close( )
    mode_file_name = "podmode_Core/mode_" + str(i) + "h5"
    mode_file = HDF5File(mesh.mpi_comm(), mode_file_name, "w")
    mode_file.write(a,"solution")
    mode_file.close()

header = 'C_matrix_Core'
C_matrix_file_name = header + '.txt'
C_matrix_file = open(C_matrix_file_name,'w')
C_Matrix = np.zeros((N_mode,N_mode))
for i in range (0,N_mode):
        for j in range (0,N_mode):
                #C_Matrix[i][j] = assemble(DS1*sc*dot(solution[i],podmode[j])*dx)
                C_Matrix[i][j] = assemble(DS1*sc*dot(podmode[i],podmode[j])*dx)
                C_matrix_file.write('%.16g\t' % (C_Matrix[i][j]))
        C_matrix_file.write('\n')
C_matrix_file.close()


# ###########################generate and save G matrix ##############################
h_c = 2.40598e4
header ='G_matrix_Core'
G_matrix_file_name = header + '.txt'
G_matrix_file = open(G_matrix_file_name,'w')
G_Matrix = np.zeros((N_mode,N_mode))
for i in range (0,N_mode):
        for j in range (0,N_mode):
                G_Matrix[i][j] = assemble(kappa*dot(grad(podmode[i]),grad(podmode[j]))*dx) + assemble(h_c*podmode[i]*podmode[j]*ds(5))
                G_matrix_file.write('%.16g\t' % (G_Matrix[i][j]))
        G_matrix_file.write('\n')
G_matrix_file.close()

print("CG is done")

#print(assemble(dot(a,a)*dx))
#######check
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ save podmode into file~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ save podmode into file~~~~~~~~~~~~~~~~~~~~~~~


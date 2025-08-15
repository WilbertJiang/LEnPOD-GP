#include <dolfin.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <map>
#include <chrono>
#include <sstream>
#include <fstream>

#include "helpers.hpp"      // load_txt, display_setup
#include "Space.h"          // compiled UFL file with functions
#include "expressions.hpp"  // kappa, sc, DS1, source_term
#include "boundary.hpp"     // BoundaryX0, BoundaryX1,...

using namespace dolfin;

// Uncomment this line to have all of the parameters be
// printed.
//#define DEBUG

int main(int argc, char** argv) {

    dolfin::init(argc,argv);
    parameters("krylov_solver")["absolute_tolerance"]=1e-25;
    parameters("krylov_solver")["relative_tolerance"]=1e-23;
    // Prints certain settings that are available
    //std::cout << parameters.str(true) << "\n\n";
    //list_lu_solver_methods();
    auto comm = MPI_COMM_WORLD;
    int mpi_size,mpi_rank;
    MPI_Comm_size(comm, &mpi_size);
    MPI_Comm_rank(comm, &mpi_rank);
    //std::cout << "RANK: " << mpi_rank << "\tSIZE: " << mpi_size << std::endl;

    // This can be changed to a config file later, maybe
    unsigned N_STATE = 0;           // Controlling Parameter: 0 -> generate data; 1 -> read previously generated data
    double l = 0.0286;               // Length         
    double w = 0.0285;               // Width
    double h = 7.2e-4;             // Height
    unsigned ls = 675;              // Meshes in the length axis]
    unsigned ws = 673;              // Meshes in the width axis
    unsigned hs = 17;               // Meshes in the height axis
    double T = 3*3000*5.33e-6;         // Final Time or Total Time (S)
    unsigned num_steps = 3000;      // Total number of time steps
    double t = 0;                 
    double delt = T/num_steps;      // Time step size

    double h_c = 2.40598e4;     
    double Ta = 0.0;                // Initial/Ambient Temperature
    unsigned Nu = 404;               // Number of functional units
    double thick_actl = 1.35e-4;    // Thickness of active layer
    double chip_area = l*w;         // Area of the chip.
    bool status;                    // flag for checking status of things

    // Load files as arrays of rows of doubles
    //std::cout << "LOADING DATA" << std::endl;
    std::string ptrace_file = "../../power_GV100_pre1_check.txt";
    std::string floorplan_file = "../../gv100_flp.txt";
    std::vector<std::vector<double>> pd, flp;
    status = Helpers::load_txt(ptrace_file, pd);
    if( !status ) {
        std::cout << "ERROR loading ptrace file: \"" << ptrace_file << "\"" << std::endl;
        return 1;
    }
    status = Helpers::load_txt(floorplan_file, flp);
    if( !status ) {
        std::cout << "ERROR loading floorplan file: \"" << floorplan_file << "\"" << std::endl;
        return 1;
    }

#ifdef DEBUG
    std::cout << std::fixed << std::setprecision(16);
    std::cout << "l\t\t" << l << std::endl;
    std::cout << "w\t\t" << w << std::endl;
    std::cout << "h\t\t" << h << std::endl;
    std::cout << "ls\t\t" << ls << std::endl;
    std::cout << "ws\t\t" << ws << std::endl;
    std::cout << "hs\t\t" << hs << std::endl;
    std::cout << "T\t\t" << T << std::endl;
    std::cout << "num_steps\t" << num_steps << std::endl;
    std::cout << "t\t\t" << t << std::endl;
    std::cout << "dt\t\t" << delt << std::endl;
    std::cout << "h_c\t\t" << h_c << std::endl;
    std::cout << "Nu\t\t" << Nu << std::endl;
    std::cout << "thick_actl\t" << thick_actl << std::endl;
    std::cout << "chip_area\t" << chip_area << std::endl;
#endif

    // Compute Power Density
    for( unsigned i = 0; i < num_steps; i++ ) {
        for( unsigned j = 0; j < Nu; j++ ) {
            pd[i][j] = 1.9*pd[i][j]/(flp[j][0]*flp[j][1]*thick_actl);
        }
    }

    // END OF INITIALIZATION PHASE
   //std::cout << "here is okay"<<std::endl;

    // set up geometric model, mesh, and function space
    std::shared_ptr<BoxMesh> mesh = 
        std::make_shared<BoxMesh>(
                BoxMesh(Point(0,0,0), Point(l,w,h), ls-1,ws-1,hs-1)
		);
    std::cout << "NUM CELLS IN MESH: " << mesh->num_cells() << std::endl;


    unsigned counter = 0;
    double tx, ty, tz;

    auto V = std::make_shared<Space::FunctionSpace>( mesh );
    std::cout << "FUNCTION SPACE DIMS: " << V->dim() << std::endl;
    
    double tol = 1E-14;
    // define thermal conductivity
    double k_0 = 100.0, k_1 = 1.2;
    // define Density
    double D0 = 2.33e3, D1 = 2.65e3;
    // define Specific Heat
    double c0 = 751.1, c1 = 680;
    Helpers::display_setup(tol,k_0,k_1,D0,D1,c0,c1);
    
    std::shared_ptr<KappaExpression> kappa = std::make_shared<KappaExpression>();
    kappa->setParams(tol,k_0,k_1);
    std::shared_ptr<DS1Expression> DS1 = std::make_shared<DS1Expression>();
    DS1->setParams(tol,D0,D1);
    std::shared_ptr<SCExpression> sc = std::make_shared<SCExpression>();
    sc->setParams(tol,c0,c1);

    std::shared_ptr<MeshFunction<size_t>> boundary_markers =
        std::make_shared<MeshFunction<size_t>>(mesh, 2);
    boundary_markers->set_all(9999);
    
    // setup boundary objects
    std::shared_ptr<BoundaryX0> bx0 = std::make_shared<BoundaryX0>();
    bx0->setLwh(l,w,h);
    std::shared_ptr<BoundaryX1> bx1 = std::make_shared<BoundaryX1>();
    bx1->setLwh(l,w,h);
    std::shared_ptr<BoundaryY0> by0 = std::make_shared<BoundaryY0>();
    by0->setLwh(l,w,h);
    std::shared_ptr<BoundaryY1> by1 = std::make_shared<BoundaryY1>();
    by1->setLwh(l,w,h);
    std::shared_ptr<BoundaryZ0> bz0 = std::make_shared<BoundaryZ0>();
    bz0->setLwh(l,w,h);
    std::shared_ptr<BoundaryZ1> bz1 = std::make_shared<BoundaryZ1>();
    bz1->setLwh(l,w,h);

    bx0->mark(*boundary_markers, 0);
    bx1->mark(*boundary_markers, 1);
    by0->mark(*boundary_markers, 2);
    by1->mark(*boundary_markers, 3);
    bz0->mark(*boundary_markers, 4);
    bz1->mark(*boundary_markers, 5);

    // END OF BOUNDARY PHASE


    std::shared_ptr<SourceTerm> f = std::make_shared<SourceTerm>();
    f->setFlp(flp); f->setPd(pd);
    f->setParams(0,h,thick_actl,Nu);
    
    auto u0 = std::make_shared<Constant>(0.0);
    auto u_n = std::make_shared<Function>(V);
    u_n->interpolate(*u0);

    // for using Dirichlet BCs    
    //auto dcval = std::make_shared<Constant>(10.0);
    //DirichletBoundary boundary;
    //DirichletBC bc(V, dcval, bz1);

    std::string solution_file_name = "";

    auto dt_ptr = std::make_shared<Constant>(delt);
    auto r = std::make_shared<Constant>(h_c);
    auto s = std::make_shared<Constant>(Ta);
    auto g0 = std::make_shared<Constant>(0.0);
    auto g1 = std::make_shared<Constant>(0.0);
    auto g2 = std::make_shared<Constant>(0.0);
    auto g3 = std::make_shared<Constant>(0.0);
    auto g4 = std::make_shared<Constant>(0.0);

    Space::LinearForm L(V);
    Space::BilinearForm a(V,V);

    L.sc = sc; L.DS1 = DS1; L.u_n = u_n;
    L.dt_in = dt_ptr; 
    L.f = f;
    //L.g0 = g0; L.g1 = g1; L.g2 = g2; L.g3 = g3; L.g4 = g4;
    //L.s = s; L.r = r;
    L.ds = boundary_markers;
    //L.dx = boundary_markers_dx;
    //L.set_cell_domains(boundary_markers_dx);

    a.sc = sc; a.DS1 = DS1; 
    a.dt_in = dt_ptr;
    a.kappa = kappa; 
    a.r = r;
    a.ds = boundary_markers;
    //a.dx = boundary_markers_dx;
    //a.set_cell_domains(boundary_markers_dx);

    Function u(V);

    Parameters solver_params;
    solver_params.add("linear_solver","gmres");
#ifdef DEBUG
    solver_params.add("print_rhs",true);
    solver_params.add("print_matrix",true);
#endif

    std::stringstream ss;
    unsigned dd = 0;
    for(unsigned i = 0; i < num_steps; i++ ) {
        std::cout << i << std::endl;
	    //if(i>99){break;}
#ifdef DEBUG
        f->printParams();
        sc->printParams();
        kappa->printParams();
        DS1->printParams();
        std::cout << "OTHER VALUES-----------" << std::endl;
        std::cout << "t:\t\t" << t << std::endl;
        std::cout << "r:\t\t" << double(*r) << std::endl;
        std::cout << "dt:\t\t" << double(*dt_ptr) << std::endl;
#endif
        auto solve_start = std::chrono::high_resolution_clock::now();
        solve( a == L,u, solver_params); 
        auto solve_stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> solve_elapsed = solve_stop-solve_start;
        std::cout << solve_elapsed.count() << std::endl;
        // update values
        // std::cout << "UPDATING VALUES FOR NEXT RUN" << std::endl;
        // save solution file 
        ss << i;
        solution_file_name = "../../solution_gv100_pre1_check/file_" + ss.str() + "h5";
        auto solution_file = HDF5File(mesh->mpi_comm(), solution_file_name, "w");
        solution_file.write(u, "solution");
        solution_file.close();

        t += delt;
        dd += 1;
        f->updateDD(dd);
        L.f = f;     

        *u_n = u;

        ss.str("");

    } // end solver loop

    return 0;

}

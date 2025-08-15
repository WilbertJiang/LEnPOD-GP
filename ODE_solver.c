/* import C, G, P_vector and temperatu:re gradient to solve ODE.  And then save the coefficient of ppodmode*/
#include "petscmat.h"
#include <slepceps.h>
#include <stdio.h>
#include <petscts.h>
#include <petscksp.h>
typedef struct {
  
   Mat         A;                 /* RHS mat, used with IFunction interface */
   Vec  vp;
  } AppCtx;
  
  // incluce user defined routin
 
  extern  PetscErrorCode   RHSFunctionHeat(TS ts,PetscReal t,Vec X,Vec r,void *ctx);
  int main(int argc,char **argv)
 {

// define the petsc parameter

Mat   CP,GP,PP,PSP,RC,CG,CPSP; // context of eps  
Vec              u,u0; // eigenvector
MPI_Comm        comm;
PetscInt        *dnnz,*onnz,i,j; // varible will be used in slepc
	 /*****************************************************************************/
IS             perm,iperm;
MatFactorInfo  info;
SlepcInitialize(&argc,&argv,0,0);
int size, rank;
MPI_Comm_size(MPI_COMM_WORLD,&size);
MPI_Comm_rank(MPI_COMM_WORLD,&rank);
comm = MPI_COMM_WORLD;
/****************************************************/ 
 //define parameter of ODE solver
 /**********************************************
  * NT: is the total time step
  * NU: the number of functional unit
  * NPOD: the maximum of podmode used to predict temperature.
  * NC: the size of C and G matrix
  */
 int NT = 3000, NC = 50, NP = 10, N_index = 0;
 N_index = atoi(argv[1]);
 printf("The index of functional unit is: %d\n",N_index);
 
/***********************************************************
 * sampling_interval: the size of sampling.
 *
 */
double sampling_interval = 3*5.33e-6; 
/***********************************************************
 *
 * malloc memory for the C matrix and read it.
 
 * ***********************************************************/
double** C = malloc(NC*sizeof(double*));
if (!C)
	return -1;
for (int i =0; i < NC; i++){
	C[i] = malloc(NC*sizeof(double));
	if (!C[i])
		return -1;
}
char nameC[200];
sprintf(nameC, "./PODmode/C_matrix_%d.txt", N_index);
FILE* filec = fopen(nameC,"r");
	for (int i = 0; i < NC; i++){
		for(int j = 0; j < NC; j++){
			C[i][j] = 0;
			if (!fscanf(filec, "%lf", &C[i][j]))
                     		break;
		}
	}
fclose(filec);
/***********************************************************
 *
 * malloc memory for the G matrix and read it.

 * ***********************************************************/
double** Gmatrix = malloc (NC*sizeof(double*));
	if (!Gmatrix)
		return -1;
	for (int i = 0; i < NC; i++){
		Gmatrix[i] = malloc(NC*sizeof(double));
		if (!Gmatrix[i])
			return -1;
	}

char nameG[200];
sprintf(nameG, "./PODmode/G_matrix_%d.txt", N_index);
FILE* fileG = fopen(nameG,"r");
for (int i = 0; i < NC; i++){
	for(j = 0; j < NC; j++){
		Gmatrix[i][j] = 0;
		if (!fscanf(fileG, "%lf", &Gmatrix[i][j]))
                     		break;
	}
}
fclose(fileG);

/***********************************************************
 *
 * malloc memory for the floorplan and read it.

 * ***********************************************************/
double **PSmatrix = malloc(NP*sizeof(double*));
if (!PSmatrix)
        return -1;
for(int i = 0; i < NP; i++){
        PSmatrix[i] = malloc(NT*sizeof(double));
        if (!PSmatrix[i])
                   return -1;
}



char namep[200];
sprintf(namep, "./Power_pre1_result/P_matrix_pre1_%d.txt", N_index);
FILE* filep = fopen(namep,"r");
for (int i = 0; i < NP; i++){
	for (int j = 0; j < NT; j++){
		PSmatrix[i][j]=0;
		if (!fscanf(filep, "%lf", &PSmatrix[i][j])) 
                     		break;
	}
}
fclose(filep);
/***********************************************************
 *
 * malloc memory for the p_lib  and read it.

 * ***********************************************************/
double** CU=malloc(NT*sizeof(double*));
	  if (!CU)
		return -1;
for(int i=0;i<NT;++i){
	CU[i] = malloc(NT*sizeof(double));
	if (!CU[i])
		return -1;
}

	//int num_pod = NP;
	for(int num_pod = 1; num_pod < NP; num_pod++){
	PetscMalloc2(num_pod,&dnnz,num_pod,&onnz); // allocate 2 arrays of matrix
    	MatCreateDense(comm,num_pod,num_pod,num_pod,num_pod,NULL,&CP);
    	MatSetUp(CP);
    	PetscFree2(dnnz,onnz);
    	MatSetOption(CP,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);
    	for (i=0; i<num_pod; i++) {
	   	for(j=0; j<num_pod;j++){
             	 MatSetValue(CP,i,j,C[i][j],INSERT_VALUES);
     
           	}
    	}
		MatAssemblyBegin(CP,MAT_FINAL_ASSEMBLY);
		MatAssemblyEnd(CP,MAT_FINAL_ASSEMBLY);
		//FREE C 
		
		
		  MatGetOrdering(CP,MATORDERINGRCM,&perm,&iperm);
		   MatFactorInfoInitialize(&info);
          info.fill          = 1.0;
          MatLUFactor(CP,perm,iperm,&info);
 
//printf("the %d th step is ok\n", num_pod);
//inverse matrix:
        PetscMalloc2(num_pod,&dnnz,num_pod,&onnz); // allocate 2 arrays of matrix
    	MatCreateDense(comm,num_pod,num_pod,num_pod,num_pod,NULL,&RC); // RC is inverse matrix of C matrix
    	MatSetUp(RC);
    	PetscFree2(dnnz,onnz);
    	MatSetOption(RC,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);
    	for (i=0; i<num_pod; i++) {
	   	for(j=0; j<num_pod;j++){
			if(i==j)
             	 MatSetValue(RC,i,j,1,INSERT_VALUES);
			 else
				 MatSetValue(RC,i,j,0,INSERT_VALUES);
     
           	}
    	}
		MatAssemblyBegin(RC,MAT_FINAL_ASSEMBLY);
		MatAssemblyEnd(RC,MAT_FINAL_ASSEMBLY);
		////////////////////////////////////////////////////////
		PetscMalloc2(num_pod,&dnnz,num_pod,&onnz); // allocate 2 arrays of matrix
    	MatCreateDense(comm,num_pod,num_pod,num_pod,num_pod,NULL,&PP);
    	MatSetUp(PP);
    	PetscFree2(dnnz,onnz);
    	MatSetOption(PP,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);
    	for (i=0; i<num_pod; i++) {
	   	for(j=0; j<num_pod;j++){
			if(i==j)
             	 MatSetValue(PP,i,j,1,INSERT_VALUES);
			 else
				 MatSetValue(PP,i,j,0,INSERT_VALUES);
     
           	}
    	}
		MatAssemblyBegin(PP,MAT_FINAL_ASSEMBLY);
		MatAssemblyEnd(PP,MAT_FINAL_ASSEMBLY);
        MatMatSolve(CP,PP,RC);
	
/*-----------------------------------------------------------------------------------------------------
save G matrix in petsc format GP
-------------------------------------------------------------------------------------------------------*/
		PetscMalloc2(num_pod,&dnnz,num_pod,&onnz); // allocate 2 arrays of matrix
    	MatCreateDense(comm,num_pod,num_pod,num_pod,num_pod,NULL,&GP);
    	MatSetUp(GP);
    	PetscFree2(dnnz,onnz);
    	MatSetOption(GP,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);
    	for (i=0; i<num_pod; i++) {
	   	for(j=0; j<num_pod;j++){
             	 MatSetValue(GP,i,j,-Gmatrix[i][j],INSERT_VALUES);
     
           	}
    	}
		MatAssemblyBegin(GP,MAT_FINAL_ASSEMBLY);
		MatAssemblyEnd(GP,MAT_FINAL_ASSEMBLY);
		//free Gmatrix
		
	/*------------------------------------------------------------------------------------------------------
	
	compute C inverse matrix times G matrix
	
	---------------------------------------------------------------------------------------------------------*/
	MatMatMult(RC,GP,MAT_INITIAL_MATRIX,1,&CG);




/*------------------------------------------------------------------------------------------------------
save source term in petsc format matrix PSP
---------------------------------------------------------------------------------------------------------*/

		PetscMalloc2(num_pod,&dnnz,NT,&onnz); // allocate 2 arrays of matrix
    	MatCreateDense(comm,num_pod,NT,num_pod,NT,NULL,&PSP);
    	MatSetUp(PSP);
    	PetscFree2(dnnz,onnz);
    	MatSetOption(PSP,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);
    
   
		for (i=0; i<num_pod; i++) {
			for(j=0; j<NT;j++){
             	 MatSetValue(PSP,i,j,PSmatrix[i][j],INSERT_VALUES); // psp is transpose matrix of PSmatrix
     
           	}
    	}
		MatAssemblyBegin(PSP,MAT_FINAL_ASSEMBLY);
		MatAssemblyEnd(PSP,MAT_FINAL_ASSEMBLY);

       MatMatMult(RC,PSP,MAT_INITIAL_MATRIX,1,&CPSP);
 
 /*-----------------------------------------------------------------------------------------------------
save mat4 matrix in petsc format PHI
-------------------------------------------------------------------------------------------------------*/

	
	
	/*-------------------------------------------------------------------------------------------
SLOVE THE EQUATION
-------------------------------------------------------------------------------------------------*/	

 
 
  
   AppCtx         appctx;                 /* user-defined application context */
   TS             ts;                     /* timestepping context */
  // Vec            u,u0;                      /* approximate solution vector */
   PetscInt       time_steps_max = 100;   /* default max timesteps */
   
   PetscReal      dt;
   VecCreate(PETSC_COMM_WORLD,&u);
    VecSetSizes(u,PETSC_DECIDE,num_pod);
	VecCreate(PETSC_COMM_WORLD,&u0);
    VecSetSizes(u0,PETSC_DECIDE,num_pod);
	VecCreate(PETSC_COMM_WORLD,&appctx.vp);
    VecSetSizes(appctx.vp,PETSC_DECIDE,num_pod);
	 VecSetType(appctx.vp, VECMPI);
	  VecSetType(u, VECMPI);
	 VecSetType(u0, VECMPI);
   	//VecDuplicate(u,&appctx.vp);
  	//VecDuplicate(xr,&u);
   //	VecDuplicate(u,&u0);
  	
	VecZeroEntries(u);
  	VecZeroEntries(u0);
   	VecZeroEntries(appctx.vp);
   // set relevant parameter
   
// create timestepping solver context

  MatCreate(PETSC_COMM_SELF,&appctx.A);
  MatSetSizes(appctx.A,PETSC_DECIDE,PETSC_DECIDE,num_pod,num_pod);
   MatSetFromOptions(appctx.A);
   MatSetUp(appctx.A);
  MatDuplicate(CG, MAT_COPY_VALUES,&appctx.A);
   
   for (int i = 0; i < NT; i ++){

 TSCreate(PETSC_COMM_SELF,&ts);
 TSSetProblemType(ts,TS_LINEAR);
 TSSetType(ts,TSRK);
 TSRKSetType( ts,TSRK4);
 TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP );
 TSSetRHSJacobian(ts,CG,CG,TSComputeRHSJacobianConstant,&appctx);
//printf("this works\n");

	

   TSSetMaxSteps(ts,time_steps_max);
   TSSetMaxTime(ts,sampling_interval);
   
  // set solution vector and initial timestepping
 dt = sampling_interval; /* which should be change to sampling interval /20 in future */
  TSSetTimeStep(ts,dt);
  
 
	MatGetColumnVector(CPSP,appctx.vp,i);
		
		//VecCopy(vp,u);
 TSSetRHSFunction(ts,NULL,RHSFunctionHeat,&appctx);
   /*
   Run the timestepping solver
   
   */
   // VecZeroEntries(u);
  TSSetSolution(ts,u0);
   TSSolve(ts,u);
    VecZeroEntries(u0);
   VecCopy(u,u0);
   for( int k = 0; k < num_pod; k++)
	{

          VecGetValues(u,1,&k,&CU[k][i]);
         

	}
   }

	char namecu[200];
	sprintf(namecu, "./Power_pre1_result/pod_result/CU_%d/CU%d.txt",N_index,num_pod);
	FILE *fileMAT4 = fopen(namecu,"w");

	for(int j = 0; j <  NT; j++){
		for (int i = 0; i < num_pod; i++){
			fprintf(fileMAT4,"%.16lg\t",CU[i][j]);
		}
		fprintf(fileMAT4,"\n");
	}
	fclose(fileMAT4);	
	TSDestroy(&ts);
	MatDestroy(&CP);
	MatDestroy(&RC);
	MatDestroy(&PP);
	MatDestroy(&GP);
	MatDestroy(&CG);
	MatDestroy(&PSP);
	MatDestroy(&CPSP);
	MatDestroy(&appctx.A);
	VecDestroy(&u);
	VecDestroy(&u0);
	VecDestroy(&appctx.vp);
	}
// free memory
for (int i= 0 ; i < NC; i++){
	free(C[i]);
}
free(C);

for (int i= 0 ; i < NC; i++){
	free(Gmatrix[i]);
}
free(Gmatrix);


for (int i= 0 ; i < NP; i++){
        free(PSmatrix[i]);
}
free(PSmatrix);

for (int i= 0 ; i < NT; i++){
        free(CU[i]);
}
free(CU);

}

PetscErrorCode  RHSFunctionHeat(TS ts,PetscReal t,Vec X,Vec r,void *ctx)
 {
   AppCtx         *appctx = (AppCtx*)ctx;     /* user-defined application context */

   MatMult(appctx -> A,X,r);
   VecAXPY(r,1.0,appctx -> vp);
  
   return 0;
 }


#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include "mesh2D.h"
#include "ellipticTri2D.h"

#define UXID 0
#define UYID 1
#define PRID 2

typedef struct {

mesh_t *mesh;
solver_t *vSolver;
solver_t *pSolver;

char *pSolverOptions, *vSolverOptions; 	

// INS SOLVER OCCA VARIABLES
dfloat rho, nu ;
iint NVfields, NTfields, Nfields;
iint NtotalDofs, NDofs; // Total DOFs for Velocity i.e. Nelements + Nelements_halo
iint ExplicitOrder; 
//
iint PID, PIID;
//
dfloat dt; // time step
dfloat lambda; // helmhotz solver -lap(u) + lamda u
dfloat finalTime; // final time to run acoustics to
iint   NtimeSteps;// number of time steps 
iint   errorStep; 

dfloat a0, a1, a2, b0, b1, b2, g0, tau; 
dfloat *NU, *NV, *rhsU, *rhsV, *rhsP;
dfloat *U, *V, *P, *PI; 
dfloat *UO, *NO;  // Storage for old data
dfloat g[2];      // gravitational Acceleration

occa::memory o_U, o_V, o_P, o_PI;
occa::memory o_NU, o_NV, o_rhsU, o_rhsV, o_rhsP; 
occa::memory o_UO, o_NO;


occa::memory o_velHaloBuffer, o_prHaloBuffer, o_totHaloBuffer; 


occa::kernel helmholtzHaloExtractKernel;
occa::kernel helmholtzHaloScatterKernel;
occa::kernel poissonHaloExtractKernel;
occa::kernel poissonHaloScatterKernel;
occa::kernel updateHaloExtractKernel;
occa::kernel updateHaloScatterKernel;

occa::kernel advectionVolumeKernel;
occa::kernel advectionSurfaceKernel;

occa::kernel advectionCubatureVolumeKernel;
occa::kernel advectionCubatureSurfaceKernel;
//
occa::kernel gradientVolumeKernel;
occa::kernel gradientSurfaceKernel;
occa::kernel divergenceVolumeKernel;
occa::kernel divergenceSurfaceKernel;
//
occa::kernel helmholtzRhsForcingKernel;
occa::kernel helmholtzRhsIpdgBCKernel;

occa::kernel poissonRhsForcingKernel;
occa::kernel poissonRhsIpdgBCKernel;

// occa::kernel updateVolumeKernel;
// occa::kernel updateSurfaceKernel;
occa::kernel updateUpdateKernel;

  
}ins_t;


ins_t *insSetup2D(mesh2D *mesh, char *options, char *velSolverOptions, char *prSolverOptions);

void insMakePeriodic2D(mesh2D *mesh, dfloat xper, dfloat yper);

void insRun2D(ins_t *solver, char *options);
void insPlotVTU2D(ins_t *solver, char *fileNameBase);
void insReport2D(ins_t *solver, iint tstep, char *options);
void insError2D(ins_t *solver, dfloat time, char *options);



void insAdvectionStep2D(ins_t *solver, iint tstep, iint haloBytes,
	                   dfloat * sendBuffer, dfloat *recvBuffer, char * options);

void insHelmholtzStep2D(ins_t *solver, iint tstep, iint haloBytes,
	                   dfloat * sendBuffer, dfloat *recvBuffer, char * options);

void insPoissonStep2D(ins_t *solver, iint tstep, iint haloBytes,
	                   dfloat * sendBuffer, dfloat *recvBuffer, char * options);


void insUpdateStep2D(ins_t *solver, iint tstep, iint haloBytes,
	                   dfloat * sendBuffer, dfloat *recvBuffer, char * options);


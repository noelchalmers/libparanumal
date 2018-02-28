
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
  precon_t *precon;

  // INS SOLVER OCCA VARIABLES
  dfloat rho, nu;
  iint NVfields, NTfields, Nfields;
  iint NtotalDofs, NDofs; // Total DOFs for Velocity i.e. Nelements + Nelements_halo
  iint ExplicitOrder; 

  dfloat dt;          // time step
  dfloat lambda;      // helmhotz solver -lap(u) + lamda u
  dfloat finalTime;   // final time to run acoustics to
  iint   NtimeSteps;  // number of time steps 
  iint   Nstages;     // Number of history states to store
  iint   index;       // Index of current state
  iint   errorStep; 


  dfloat a0, a1, a2, b0, b1, b2, c0, c1, c2, g0, tau; 
  dfloat *rhsU, *rhsV, *rhsP;
  dfloat *U, *V, *P; 

  dfloat dtfactor;

  dfloat *NU, *NV;
  dfloat *Px, *Py;
  dfloat *PI;

  dfloat g[2];      // gravitational Acceleration

  iint Nsubsteps;  
  dfloat *Ud, *Vd, *Ue, *Ve, *resU, *resV, sdt;
  occa::memory o_Ud, o_Vd, o_Ue, o_Ve, o_resU, o_resV;

  occa::kernel subCycleVolumeKernel,  subCycleCubatureVolumeKernel ;
  occa::kernel subCycleSurfaceKernel, subCycleCubatureSurfaceKernel;;
  occa::kernel subCycleRKUpdateKernel;
  occa::kernel subCycleExtKernel;

 
  occa::memory o_U, o_V, o_P;
  occa::memory o_rhsU, o_rhsV, o_rhsP; 

  occa::memory o_NU, o_NV;
  occa::memory o_Px, o_Py;

  occa::memory o_UH, o_VH;
  occa::memory o_PI, o_PIx, o_PIy;

  occa::memory o_vHaloBuffer, o_pHaloBuffer, o_tHaloBuffer; 

  occa::kernel totalHaloExtractKernel;
  occa::kernel totalHaloScatterKernel;

  occa::kernel velocityHaloExtractKernel;
  occa::kernel velocityHaloScatterKernel;
  occa::kernel pressureHaloExtractKernel;
  occa::kernel pressureHaloScatterKernel;

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
  occa::kernel poissonPenaltyKernel;
  
  occa::kernel updateUpdateKernel;

}ins_t;

typedef struct{

  iint row;
  iint col;
  iint ownerRank;
  dfloat val;

} nonZero_t;



ins_t *insSetup2D(mesh2D *mesh, iint i, char *options, char *velSolverOptions, char *prSolverOptions, char *bdryHeaderFileName);

void insMakePeriodic2D(mesh2D *mesh, dfloat xper, dfloat yper);

void insRun2D(ins_t *solver, char *options);
void insPlotVTU2D(ins_t *solver, char *fileNameBase);
void insReport2D(ins_t *solver, iint tstep, char *options);
void insError2D(ins_t *solver, dfloat time, char *options);
void insErrorNorms2D(ins_t *solver, dfloat time, char *options);

void insAdvectionStep2D(ins_t *solver, iint tstep, iint haloBytes,
	                   dfloat * sendBuffer, dfloat *recvBuffer, char * options);

void insHelmholtzStep2D(ins_t *solver, iint tstep, iint haloBytes,
	                   dfloat * sendBuffer, dfloat *recvBuffer, char * options);

void insPoissonStep2D(ins_t *solver, iint tstep, iint haloBytes,
	                   dfloat * sendBuffer, dfloat *recvBuffer, char * options);


void insUpdateStep2D(ins_t *solver, iint tstep, iint haloBytes,
	                   dfloat * sendBuffer, dfloat *recvBuffer, char * options);



void insAdvectionSubCycleStep2D(ins_t *solver, iint tstep,
                     dfloat * tsendBuffer, dfloat *trecvBuffer, 
                     dfloat * sendBuffer, dfloat *recvBuffer,char * options);



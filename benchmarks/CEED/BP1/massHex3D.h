#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "mesh3D.h"

typedef struct {

  occa::memory o_vmapPP;
  occa::memory o_faceNodesP;

  occa::memory o_oasForward;
  occa::memory o_oasBack;
  occa::memory o_oasDiagInvOp;
  occa::memory o_invDegreeP;

  occa::memory o_oasForwardDg;
  occa::memory o_oasBackDg;
  occa::memory o_oasDiagInvOpDg;
  occa::memory o_invDegreeDGP;
  
  occa::kernel restrictKernel;
  occa::kernel preconKernel;

  occa::kernel coarsenKernel;
  occa::kernel prolongateKernel;  

  
  ogs_t *ogsP, *ogsDg;

  occa::memory o_diagA;

  // coarse grid basis for preconditioning
  occa::memory o_V1, o_Vr1, o_Vs1, o_Vt1;
  occa::memory o_r1, o_z1;
  dfloat *r1, *z1;
  void *xxt, *amg, *almond;

  occa::memory o_coarseInvDegree;
  occa::memory o_ztmp;

  int coarseNp;
  int coarseTotal;
  int *coarseOffsets;
  dfloat *B, *tmp2;
  occa::memory *o_B, o_tmp2;
  void *xxt2;
  void *parAlmond;

  
} precon_t;

void massRunHex3D(mesh3D *mesh);

void massOccaRunHex3D(mesh3D *mesh);

void massSetupHex3D(mesh3D *mesh, occa::kernelInfo &kernelInfo);

void massVolumeHex3D(mesh3D *mesh);

void massSurfaceHex3D(mesh3D *mesh, dfloat time);

void massUpdateHex3D(mesh3D *mesh, dfloat rka, dfloat rkb);

void massErrorHex3D(mesh3D *mesh, dfloat time);

void massParallelGatherScatter(mesh3D *mesh, ogs_t *ogs, occa::memory &o_v, occa::memory &o_gsv,
				    const char *type, const char *op);

precon_t *massPreconditionerSetupHex3D(mesh3D *mesh, ogs_t *ogs, dfloat lambda, const char *options);

void massCoarsePreconditionerHex3D(mesh_t *mesh, precon_t *precon, dfloat *x, dfloat *b);

void massCoarsePreconditionerSetupHex3D(mesh_t *mesh, precon_t *precon, ogs_t *ogs, dfloat lambda, const char *options);

typedef struct {

  mesh_t *mesh;

  precon_t *precon;

  ogs_t *ogs;

  ogs_t *ogsDg;

  // C0 halo gather-scatter info
  ogs_t *halo;

  // C0 nonhalo gather-scatter info
  ogs_t *nonHalo;
  
  
  int Nblock;
  
  occa::memory o_p; // search direction
  occa::memory o_z; // preconditioner solution
  occa::memory o_zP; // extended OAS preconditioner patch solution
  occa::memory o_Ax; // A*initial guess
  occa::memory o_Ap; // A*search direction
  occa::memory o_tmp; // temporary
  occa::memory o_grad; // temporary gradient storage (part of A*)
  occa::memory o_rtmp;
  occa::memory o_invDegree;
  occa::memory o_pAp;

  // pipelining CG
  occa::memory o_Aw;
  occa::memory o_w;
  occa::memory o_s; 

  
  dfloat *sendBuffer, *recvBuffer;

  // HOST shadow copies
  dfloat *Ax, *p, *r, *z, *zP, *Ap, *tmp, *grad;

  // integration storage for BP3
  int gNq;
  occa::memory o_gjGeo; // Jacobian matrix at integration nodes
  occa::memory o_gjI;    // interpolate from GLL to integration nodes
  occa::memory o_gjD;    // differentiate and interpolate from GLL to integration nodes
  occa::memory o_gjD2;  // differentiate from GJ to GJ nodes

  // list of elements that are needed for global gather-scatter
  int NglobalGatherElements;
  int *globalGatherElementList;
  occa::memory o_globalGatherElementList;

  // list of elements that are not needed for global gather-scatter
  int NlocalGatherElements;
  int *localGatherElementList;
  occa::memory o_localGatherElementList;
  
  occa::kernel AxKernel;
  occa::kernel partialAxKernel;
  
  occa::kernel gradientKernel;
  occa::kernel partialGradientKernel;

  occa::kernel ipdgKernel;
  occa::kernel partialIpdgKernel;
  
  occa::stream defaultStream;
  occa::stream dataStream;

  occa::kernel combinedInnerProductKernel;
  occa::kernel combinedUpdateKernel;
  
}solver_t;

// block size for reduction (hard coded)
#define blockSize 1024

void massMatrixFreeAx(void **args, occa::memory o_q, occa::memory o_Aq, const char* options);

int massSolveHex3D(solver_t *solver, dfloat lambda, occa::memory &o_r, occa::memory &o_x, int maxIterations, const char *options);

solver_t *massSolveSetupHex3D(mesh_t *mesh, dfloat lambda, occa::kernelInfo &kernelInfo, const char *options);


void massStartHaloExchange3D(solver_t *solver, occa::memory &o_q, dfloat *sendBuffer, dfloat *recvBuffer);

void massInterimHaloExchange3D(solver_t *solver, occa::memory &o_q, dfloat *sendBuffer, dfloat *recvBuffer);

void massEndHaloExchange3D(solver_t *solver, occa::memory &o_q, dfloat *recvBuffer);

void massParallelGatherScatterHex3D(mesh3D *mesh, ogs_t *ogs, occa::memory &o_q, occa::memory &o_gsq, const char *type, const char *op);

void massHaloGatherScatter(solver_t *solver, 
			       ogs_t *halo, 
			       occa::memory &o_v,
			       const char *type,
			       const char *op);

void massNonHaloGatherScatter(solver_t *solver, 
				  ogs_t *nonHalo, 
				  occa::memory &o_v,
				  const char *type,
				  const char *op);


void massParallelGatherScatterSetup(mesh_t *mesh,    // provides DEVICE
					int Nlocal,     // number of local nodes
					int Nbytes,     // number of bytes per node
					int *gatherLocalIds,  // local index of nodes
					int *gatherBaseIds,   // global index of their base nodes
					int *gatherHaloFlags,
					ogs_t **halo,
					ogs_t **nonHalo);   // 1 for halo node, 0 for not
